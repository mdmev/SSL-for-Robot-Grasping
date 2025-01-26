import torch.nn as nn


class Decoder3D(nn.Module):
    def __init__(self):
        super(Decoder3D, self).__init__()
        
        self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1) # Output: [B, 64, 2, 2, 2]
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, stride=1, padding=2)  # Output: [B, 32, 4, 4, 4]
        self.conv3 = nn.Conv3d(32, 16, kernel_size=7, stride=1, padding=3)  # Output: [B, 16, 8, 8, 8]
        self.conv4 = nn.Conv3d(16, 1, kernel_size=9, stride=1, padding=4)   # Output: [B, 1, 16, 16, 16]

        self.upsample1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # Upsample to [4, 4, 4]
        self.upsample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # Upsample to [8, 8, 8]
        self.upsample3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # Upsample to [16, 16, 16]
        self.upsample4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # Upsample to [32, 32, 32]

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation(self.conv1(x))  # Conv1 + Activation
        x = self.upsample1(x)               # Upsample 1
        x = self.activation(self.conv2(x))  # Conv2 + Activation
        x = self.upsample2(x)               # Upsample 2
        x = self.activation(self.conv3(x))  # Conv3 + Activation
        x = self.upsample3(x)               # Upsample 3
        x = self.activation(self.conv4(x))  # Conv4 + Activation
        x = self.upsample4(x)               # Upsample 4
        return x
