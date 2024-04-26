#Import the libraries
import torch.nn as nn

from config import NUM_PICS
from torch import device
from torch import cuda

# Define the U-Net architecture
class UNet3D(nn.Module):
    def __init__(self, number_of_layers=2):
        super(UNet3D, self).__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")

        # Number of first and last layers' channels
        base_channels = 16

        encoder = []
        decoder = []

        # Encoder, get layers 1 more then the number of layers
        # to account for final layers = base_channels
        for i in range(number_of_layers + 1):
            in_channels = 1 if i == 0 else base_channels * (2 ** (i - 1))
            out_channels = base_channels * (2 ** i)

            encoder_layer = [
                nn.Conv3d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm3d(out_channels),
                nn.Dropout3d(0.1 + 0.05 * i),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ]

            if i < number_of_layers:
                encoder_layer.append(nn.MaxPool3d(2))
        
            encoder.extend(encoder_layer)

        for j in range(number_of_layers, -1, -1):
            in_channels = base_channels * (2 ** j)
            out_channels = base_channels * (2 ** (j - 1))

            decoder_layer = [
                nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2),
                nn.ReLU(),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
                nn.Dropout3d(0.2 - 0.05 * j),
                nn.Conv3d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU()
            ]

            decoder.extend(decoder_layer)

        # Transform to Sequential
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        # Final output layer
        self.out_conv = nn.Conv3d(base_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

    def forward(self, x):
        # Encoder
        enc_features = self.encoder(x)
        
        # Decoder
        dec_features = self.decoder(enc_features)
        
        # Final output
        out = self.out_conv(dec_features)
        out = self.sigmoid(out)
        
        return out
