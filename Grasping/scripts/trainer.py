import wandb
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# remove score for now

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.step = 0

        self.criterion_joints = nn.MSELoss()   #  MSELoss
        # self.criterion_score = nn.L1Loss()     #  MSELoss

        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        wandb.init(
            project=config.project_name,
            name="Supervised",
            config={        
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate,
                "val_steps": config.val_steps,
            }
        )

    def train(self):
        print("Initial Evaluation at Step 0")
        self.val_step()
        for epoch in range(self.config.num_epochs):
            print(f"Training: Epoch [{epoch+1}/{self.config.num_epochs}]")
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", unit="step"):
                # print(f"voxel: {voxel.shape}\npose: {pose.shape}\njoints_gt: {joints_gt.shape}\nscore gt: {score_gt.shape}")
                voxel = batch["voxel"].to(self.config.device)
                pose = batch["pose"].to(self.config.device)
                joints_gt = batch["joints"].to(self.config.device)
                score_gt = batch["score"].to(self.config.device)
                loss = self.train_step(voxel, pose, joints_gt, score_gt)
                # print(f"Training at Step [{self.step}] - Loss: {loss:.4f}")

                if self.step % self.config.val_steps == 0 and self.step > 0:
                    val_loss = self.val_step()
                    # print(f"Validation at Step [{self.step}] - Loss: {val_loss:.4f}")
                    wandb.log({"val_loss": val_loss}, step=self.step)

                wandb.log({"loss": loss}, step=self.step)

                # self.scheduler.step()
                self.step+=1
        torch.save(self.model.state_dict(), f'{self.config.output}/model.pth')
        wandb.finish()

    def train_step(self, voxel, pose, joints_gt, score_gt):
        self.model.train()
        
        if self.config.predict_stability:
            pred_joints, pred_score = self.model(voxel, pose)
            # print(f"\npred_joints: {pred_joints.shape}\npred_score: {pred_score.shape}\n\n\n\n")
            loss_joints = self.criterion_joints(pred_joints, joints_gt)
            loss_score = self.criterion_score(pred_score.squeeze(), score_gt)

            loss = self.config.alpha * loss_joints + self.config.beta * loss_score
       
       
        else:
            pred_joints = self.model(voxel, pose)
            # print(f"\npred_joints: {pred_joints.shape}")
            loss = self.criterion_joints(pred_joints, joints_gt)

           
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                voxel = batch["voxel"].to(self.config.device)
                pose = batch["pose"].to(self.config.device)
                joints_gt = batch["joints"].to(self.config.device)
                score_gt = batch["score"].to(self.config.device)


                if self.config.predict_stability:
                    pred_joints, pred_score = self.model(voxel, pose)
                    loss_joints = self.criterion_joints(pred_joints, joints_gt)
                    loss_score = self.criterion_score(pred_score.squeeze(), score_gt)
                    loss = self.config.alpha * loss_joints + self.config.beta * loss_score
                
                
                else:
                    pred_joints = self.model(voxel, pose)
                    loss = self.criterion_joints(pred_joints, joints_gt)

                val_loss += loss.item()
        avg_loss = val_loss / len(self.val_loader)
        return avg_loss

    def test(self):
        self.model.eval()
        predicted_grasps = []
        scores = []

        with torch.no_grad():
            for batch in self.test_loader:
                voxel = batch["voxel"].to(self.config.device)
                pose = batch["pose"].to(self.config.device)
                # joints_gt = batch["joints"].to(self.config.device)
                score_gt = batch["score"].to(self.config.device)

                if self.config.predict_stability:
                    pred_joints, pred_score = self.model(voxel, pose)
                else:
                    pred_joints = self.model(voxel, pose)

                pred_joints_np = pred_joints.cpu().numpy()
                pose_np = pose.cpu().numpy()
                scores_np = score_gt.cpu().numpy()
                pred_score_np = pred_score.cpu().numpy()

                for i in range(len(pred_joints_np)):
                    grasp_vector = np.concatenate([pose_np[i], pred_joints_np[i]])
                    predicted_grasps.append(grasp_vector)

                    if pred_score_np is not None:
                        scores.append(pred_score_np[i])
                    else:
                        scores.append(scores_np[i])


        predicted_grasps = np.array(predicted_grasps, dtype=np.float32)
        output_data = {"grasps": predicted_grasps}

        scores = np.array(scores, dtype=np.float32)
        output_data["scores"] = scores

        np.savez(f"{self.config.output}recording.npz", **output_data)
        print(f"Predicted grasps saved to {self.config.output}")
