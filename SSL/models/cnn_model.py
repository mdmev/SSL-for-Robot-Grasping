import torch.nn as nn

class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()

        self.conv1 = nn.Conv3d(1, 16, kernel_size=9, stride=2, padding=4)   # Output: [B, 16, 16, 16, 16]
        self.conv2 = nn.Conv3d(16, 32, kernel_size=7, stride=2, padding=3)  # Output: [B, 32, 8, 8, 8]
        self.conv3 = nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2)  # Output: [B, 64, 4, 4, 4]
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1) # Output: [B, 128, 2, 2, 2]

        self.layer_norm = nn.LayerNorm([128, 2, 2, 2])

        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.activation(self.conv1(x))  # Layer 1
        x = self.activation(self.conv2(x))  # Layer 2
        x = self.activation(self.conv3(x))  # Layer 3
        x = self.layer_norm(self.conv4(x))  # Layer 4 + Normalization
        return x
