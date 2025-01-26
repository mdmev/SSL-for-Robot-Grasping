import torch
import argparse

class Config:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.project_name = args.project_name
        self.val_steps = args.val_steps
        self.output = args.output
        self.warmup_percentage = args.warmup_percentage
        self.predict_stability = args.predict_stability
        self.weight_decay = args.weight_decay
        self.alpha = args.alpha
        self.beta = args.beta

def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D models with configurable parameters.")
    # parser.add_argument("--data_path", type=str, default="/workspace1/guillfa/adl4r/mesh2sdf/example/npy_files",
    #                     help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--project_name", type=str, default="Grasping_test",
                        help="Project name for tracking purposes.")
    parser.add_argument("--val_steps", type=int, default=1, help="Number of validation steps.")
    parser.add_argument("--output", type=str, default="runs/", help="Directory for output results.")
    parser.add_argument("--warmup_percentage", type=float, default="0", help="warmup percentage")
    parser.add_argument("--predict_stability", type=bool, default=True, help="predict stability")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_ ecay")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="beta")
    return parser.parse_args()

