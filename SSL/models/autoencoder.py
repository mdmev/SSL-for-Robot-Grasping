import torch.nn as nn
from models.cnn_model import Conv3DNet
from models.cnn_decoder import Decoder3D

class Autoencoder3D(nn.Module):
    def __init__(self):
        super(Autoencoder3D, self).__init__()
        self.encoder = Conv3DNet()  # The existing Conv3DNet serves as the encoder
        self.decoder = Decoder3D()  # The existing Decoder3D serves as the decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
