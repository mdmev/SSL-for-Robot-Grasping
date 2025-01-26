import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scripts.utils import *

class SDFDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.files = [f for f in os.listdir(config.data_path) if f.endswith('.npy')]
        
        self.file_path = os.path.join(self.config.data_path, self.files[0])
        self.sdf_data = np.load(self.file_path)
        self.sdf_tensor = torch.from_numpy(self.sdf_data).float()

        self.mask = create_block_mask([8,8,8], 0.4, 3)
        self.inverse_mask = 1 - self.mask


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        masked_sdf = apply_mask_3d(self.sdf_tensor, self.mask)
        inverse_masked_sdf = apply_mask_3d(self.sdf_tensor, self.inverse_mask)
        # print("min:", self.sdf_tensor.min().item(), "max:",self.sdf_tensor.max().item(), "mean:", self.sdf_tensor.mean().item())
        return (self.sdf_tensor, masked_sdf, inverse_masked_sdf, 
                self.mask, self.inverse_mask, self.files[0])
