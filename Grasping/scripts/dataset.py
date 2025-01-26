import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class GraspDataset(Dataset):

    def __init__(self, voxel_path, data_path, transform=None, predict_stability=True):

        # -> SDF
        super().__init__()
        self.voxel = np.load(voxel_path)  # shape (32, 32, 32)
        self.data = np.load(data_path)
        self.grasps = self.data['grasps']  # shape (N, 19)
        self.scores = self.data['scores']  # shape (N,)
        self.transform = transform
        self.predict_stability = predict_stability

        # Basic checks
        assert self.voxel.shape == (32, 32, 32), \
            f"Expected voxel of shape (32,32,32), got {self.voxel.shape}"
        assert len(self.grasps) == len(self.scores), \
            "Mismatch between number of grasps and scores!"

        # Convert to float32 for PyTorch
        self.voxel = self.voxel.astype(np.float32)

    def __len__(self):
        return len(self.grasps) # 24

    def __getitem__(self, idx):
        grasp_vec = self.grasps[idx].astype(np.float32)
        pose = grasp_vec[:7]     # shape (7,)
        joints = grasp_vec[7:]   # shape (12,)

        # Convert score
        score = self.scores[idx].astype(np.float32)
        # Optionally apply transformations to the voxel grid
        voxel_data = self.voxel.copy()
        if self.transform is not None:
            voxel_data = self.transform(voxel_data)

        # Convert everything to torch Tensors
        voxel_data = torch.from_numpy(voxel_data).unsqueeze(0)  # shape (1, 32, 32, 32)
        pose = torch.from_numpy(pose)    # shape (7,)
        joints = torch.from_numpy(joints)  # shape (12,)
        score = torch.tensor(score)        # shape ()
        sample = {
            "voxel": voxel_data,
            "pose": pose,
            "joints": joints,
            "score": score
        }

        return sample
