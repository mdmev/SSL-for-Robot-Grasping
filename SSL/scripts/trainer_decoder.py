import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from scripts.utils import *

class DecoderTrainer:
    def __init__(self, config, train_loader, val_loader, decoder_model, student_model):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.decoder_model = decoder_model
        self.student_model = student_model.eval()

        for param in self.student_model.parameters():
            param.requires_grad = False

        self.decoder_optimizer = optim.Adam(self.decoder_model.parameters(), lr=config.learning_rate) # 1e-4
        total_steps = len(self.train_loader) * self.config.num_epochs
        
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.decoder_optimizer,
            T_max=total_steps,
            eta_min=0.0
        )

        if wandb.run is not None:
            wandb.finish()
        
        wandb.init(
            project=config.project_name,
            name="decoder",
            config={
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "val_steps": config.val_steps
            }
        )

    def train(self):
        step = 0
        self.val_step(step)
        for epoch in range(self.config.num_epochs):
            print(f"Decoder Training: Epoch [{epoch+1}/{self.config.num_epochs}]")
            for inputs, masked_inputs, inverse_masked_inputs, mask, inverse_mask, path in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", unit="step"):
                # print(inputs.shape, "\n",masked_inputs.shape,"\n", mask.shape)
                
                inputs = inputs.unsqueeze(1).to(self.config.device)
                masked_inputs = masked_inputs.unsqueeze(1).to(self.config.device)
                inverse_masked_inputs = inverse_masked_inputs.to(self.config.device)
                inverse_mask = inverse_mask.to(self.config.device)
                loss = self.train_step(inputs, masked_inputs, inverse_masked_inputs, mask, inverse_mask, path, step)
                wandb.log(
                    {
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[0]
                    },
                    step=step
                    )
                # wandb.log({"loss": loss, "lr": self.scheduler.get_last_lr()[0]}, step=step)

                if step % self.config.val_steps == 0 and step > 0:
                    self.val_step(step)

                self.scheduler.step()                
                step += 1

        torch.save(self.decoder_model.state_dict(), f'{self.config.output}/decoder_model.pth')
        wandb.finish()


    def train_step(self, inputs, masked_inputs, inverse_masked_inputs, mask, inverse_mask, path, step):
        self.decoder_model.train()

        
        with torch.no_grad():
            student_outputs = self.student_model(masked_inputs)

        decoder_outputs = self.decoder_model(student_outputs)
        # masked_outputs = apply_mask_decoder(decoder_outputs, inverse_mask)

        loss = F.mse_loss(decoder_outputs, inputs)

        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.decoder_optimizer.step()
        
        return loss.item()

    def val_step(self, step):
        self.decoder_model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, masked_inputs, inverse_masked_inputs, mask, inverse_mask, path in self.val_loader:
                inputs = inputs.unsqueeze(1).to(self.config.device)
                masked_inputs = masked_inputs.unsqueeze(1).to(self.config.device)
                inverse_mask = inverse_mask.to(self.config.device)
                inverse_masked_inputs = inverse_masked_inputs.to(self.config.device)
                
                student_outputs = self.student_model(masked_inputs)

                decoder_outputs = self.decoder_model(student_outputs)
                
                # masked_outputs =  apply_mask_decoder(decoder_outputs, inverse_mask)

                val_loss += F.mse_loss(decoder_outputs, inputs).item()

        avg_loss = val_loss / len(self.val_loader)
        wandb.log({"val_loss": avg_loss}, step=step)
        print(f"Validation at Step [{step}] - Decoder Loss: {avg_loss:.4f}")
        self.decoder_model.train()