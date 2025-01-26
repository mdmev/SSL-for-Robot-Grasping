import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from scripts.dataset import GraspDataset
from scripts.model import GraspModel
from scripts.trainer import Trainer
from scripts.config import *

voxel_path = "datasets/voxel_grid_32x32x32.npy"          # shape (32, 32, 32)
data_path = "datasets/recording.npz"                     # 'grasps' (N, 19), 'scores' (N,)

# data = np.load(data_path)
# print(data)
# print(data["scores"].shape)
# print(data["grasps"].shape)
# data_2 = "runs/predictions.npz"
# data = np.load(data_2)
# print(data)
# print(data["scores"].shape)
# print(data["grasps"].shape)
# exit(1)

def main() -> None:
    args = parse_args()
    config = Config(args)

    if config.output is not None:
        os.makedirs(config.output, exist_ok=True)

    dataset = GraspDataset(voxel_path, data_path, predict_stability=True)

    train_loader = DataLoader(dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(dataset,   batch_size=24, shuffle=False)
    test_loader = DataLoader(dataset,  batch_size=24, shuffle=False)

    model = GraspModel(config.predict_stability).to(config.device)
    
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)

    trainer.train()
    trainer.test()

    
if __name__ == "__main__":
    main()