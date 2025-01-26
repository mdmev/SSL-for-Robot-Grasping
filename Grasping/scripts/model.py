import torch
import torch.nn as nn

class GraspModel(nn.Module):
    def __init__(self, predict_stability=True):
        super().__init__()
        self.predict_stability = predict_stability

        # 3D Convolutional encoder
        # Input shape: (N, 1, 32, 32, 32)


        self.conv_block = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.MaxPool3d(kernel_size=2, stride=2),  # -> (N, 32, 16, 16, 16)

            nn.Conv3d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv3d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.01),
            # nn.MaxPool3d(kernel_size=2, stride=2),  # -> (N, 128, 4, 4, 4)

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, 2, 2, 2])
        ) # -> CNN_SSL

        fc_input_dim = 128 * 2 * 2 * 2 + 7

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.joint_head = nn.Linear(512, 12)

        if self.predict_stability:
            self.stability_head = nn.Linear(512, 1)

    def forward(self, voxel, pose):
        batch_size = voxel.shape[0]

        # 3D conv
        x = self.conv_block(voxel)                   # -> (N, 128, 4, 4, 4)
        x = x.view(batch_size, -1)                   # Flatten -> (N, 8192)

        # Concatenate flattened voxel features with 7D pose
        x = torch.cat([x, pose], dim=1)              # -> (N, 8192 + 7)

        # FC layers
        x = self.fc_layers(x)                        # -> (N, 512)

        # Predict joint angles
        joints = self.joint_head(x)                  # -> (N, 12)

        if self.predict_stability:
            stability = self.stability_head(x)       # -> (N, 1)
            return joints, stability
        else:
            return joints
