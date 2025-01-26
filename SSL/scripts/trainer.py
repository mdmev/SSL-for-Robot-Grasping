import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import math

import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from scripts.utils import *

class StudentTrainer:
    def __init__(self, config, train_loader, val_loader, test_loader, student_model, teacher_model):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.student_model = student_model
            
        self.teacher_model = teacher_model.eval()

        self.student_optimizer = optim.Adam(self.student_model.parameters(), lr=config.learning_rate)

        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = total_steps*config.warmup_percentage
        cosine_steps = total_steps - warmup_steps

        def warmup_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(
            self.student_optimizer,
            lr_lambda=warmup_lambda
        )

        cosine_scheduler = CosineAnnealingLR(
            self.student_optimizer,
            T_max=cosine_steps,
            eta_min=0.0
        )

        self.scheduler = SequentialLR(
            self.student_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        if wandb.run is not None:
            wandb.finish()
        
        wandb.init(
            project=config.project_name,
            name="student",
            config={        
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "ema_decay": config.ema_decay,
                "learning_rate": config.learning_rate,
                "val_steps": config.val_steps,
                "warmup_steps": warmup_steps
            }
        )

    def train(self):
        step = 0
        
        print("Initial Evaluation at Step 0")
        self.val_step(step)
        for epoch in range(self.config.num_epochs):
            print(f"Student Training: Epoch [{epoch+1}/{self.config.num_epochs}]")
            for inputs, masked_inputs, _, _, _, _ in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", unit="step"):
                inputs = inputs.unsqueeze(1).to(self.config.device)
                masked_inputs = masked_inputs.unsqueeze(1).to(self.config.device)
                loss = self.train_step(inputs, masked_inputs, step)
                wandb.log(
                    {
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[0]
                    },
                    step=step
                    )

                if step % self.config.val_steps == 0 and step > 0:
                    self.val_step(step)

                self.scheduler.step()                
                step += 1

        torch.save(self.student_model.state_dict(), f'{self.config.output}/student_model.pth')
        torch.save(self.teacher_model.state_dict(), f'{self.config.output}/teacher_model.pth')
        wandb.finish()

    def train_step(self, inputs, masked_inputs, step):

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        student_outputs = self.student_model(masked_inputs)

        loss = F.smooth_l1_loss(student_outputs, teacher_outputs)

        self.student_optimizer.zero_grad()
        loss.backward()
        self.student_optimizer.step()
        update_teacher_weights(self.student_model, self.teacher_model, step, self.config.ema_decay)
        return loss.item()

    def val_step(self, step):
        val_loss = 0
        with torch.no_grad():
            for inputs, masked_inputs, _,_,_,_ in self.val_loader:
                inputs = inputs.unsqueeze(1).to(self.config.device)
                masked_inputs = masked_inputs.unsqueeze(1).to(self.config.device)

                teacher_outputs = self.teacher_model(inputs)
                student_outputs = self.student_model(masked_inputs)

                val_loss += F.smooth_l1_loss(student_outputs, teacher_outputs).item()

        avg_loss = val_loss / len(self.val_loader)
        wandb.log({"val_loss": avg_loss}, step=step)
        print(f"Validation at Step [{step}] - Student Loss: {avg_loss:.4f}")
        self.student_model.train()