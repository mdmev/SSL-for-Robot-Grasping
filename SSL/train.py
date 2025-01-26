import copy
import os
import numpy as np

from torch.utils.data import DataLoader, random_split

from scripts.trainer import StudentTrainer
from scripts.trainer_decoder import DecoderTrainer
from config.config import *
from scripts.dataset import SDFDataset
from models.cnn_model import Conv3DNet
from models.cnn_decoder import Decoder3D

def main() -> None:
    args = parse_args()
    print("Initializing Config...")
    config = Config(args)
    print(f"Training model on {config.device}")
    if config.output is not None:
        os.makedirs(config.output, exist_ok=True)

    dataset = SDFDataset(config)

    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    student_model = Conv3DNet().to(config.device)
    teacher_model = copy.deepcopy(student_model).to(config.device)
    decoder_model = Decoder3D().to(config.device)

    # Train Student
    student_trainer = StudentTrainer(config, train_loader, val_loader, test_loader, student_model, teacher_model)
    student_trainer.train()

    # Train Decoder
    decoder_trainer = DecoderTrainer(config, train_loader, val_loader, decoder_model, student_model)
    decoder_trainer.train()


if __name__ == "__main__":
    main()
