import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class AutoencoderTrainer:
    def __init__(self, encoder, decoder, device="cuda", lr=1e-4):
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        # Combine the encoder+decoder parameters for single optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=lr
        )
        self.criterion = nn.MSELoss()

    def train_step(self, x):
        # x shape = [B, 32, 32, 32]
        # Add channel dimension for 3D conv
        x = x.unsqueeze(1).to(self.device)  # [B,1,32,32,32]

        # Forward pass
        encoded = self.encoder(x)          # [B,1024,8,8,8]
        decoded = self.decoder(encoded)    # [B,1,32,32,32]

        # Compute reconstruction loss (MSE)
        loss = self.criterion(decoded, x)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, x):
        with torch.no_grad():
            x = x.unsqueeze(1).to(self.device)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            loss = self.criterion(decoded, x)
        return loss.item()
