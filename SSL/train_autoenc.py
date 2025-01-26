from torch.utils.data import DataLoader
from config.config import *
from scripts.dataset import SDFDataset
from models.cnn_model import Conv3DNet
from models.cnn_decoder import Decoder3D
from scripts.autoenc_trainer import AutoencoderTrainer
import os

def main():
    args = parse_args()
    
    config = Config(args)

    dataset = SDFDataset(config)

    
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    encoder = Conv3DNet()
    decoder = Decoder3D()

    trainer = AutoencoderTrainer(encoder, decoder, device=config.device, lr=1e-4)
    num_epochs = 500
    for epoch in range(num_epochs):
        train_loss = 0.0
        for _ in range(10):  
            for inputs, _, _, _, _, _ in train_loader:
                loss_val = trainer.train_step(inputs)
                train_loss += loss_val
        val_loss = 0.0
        for inputs, _, _, _, _, _ in train_loader:
            val_loss += trainer.val_step(inputs)

        avg_train_loss = train_loss / (10 * len(train_loader))
        avg_val_loss = val_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    os.makedirs("autoencoder_runs", exist_ok=True)
    torch.save(encoder.state_dict(), "autoencoder_runs/encoder.pth")
    torch.save(decoder.state_dict(), "autoencoder_runs/decoder.pth")
    print("Finished Overfitting on Single Object!")


if __name__ == "__main__":
    main()
