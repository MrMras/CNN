import torch
import torch.nn as nn
from torch import device, cuda

class UNet3D(nn.Module):
    def __init__(self):
        self.num_of_layers = 2
        super(UNet3D, self).__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        
        # Encoder
        # Layer 1
        self.enc_conv0 = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16), nn.Dropout3d(0.1))
        self.enc_conv1 = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16))
        self.pool1 = nn.MaxPool3d(2)
        # Layer 2
        self.enc_conv2 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32), nn.Dropout3d(0.1))
        self.enc_conv3 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32))
        self.pool2 = nn.MaxPool3d(2)
        # Layer 3
        self.enc_conv4 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(64), nn.Dropout3d(0.1))
        self.enc_conv5 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(64))
        
        # Decoder
        # Layer 2
        self.up1 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec_conv0 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32), nn.Dropout3d(0.1))
        self.dec_conv1 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32))
        # Layer 1
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16), nn.Dropout3d(0.1))
        self.dec_conv3 = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16))

        # Final layer
        self.final = nn.Conv3d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

        self.to(self.device)

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv0(x)
        x2 = self.enc_conv1(x1)
        p1 = self.pool1(x2)
        
        x3 = self.enc_conv2(p1)
        x4 = self.enc_conv3(x3)
        p2 = self.pool2(x4)

        x5 = self.enc_conv4(p2)
        x6 = self.enc_conv5(x5)
        
        # Decoder
        up1 = self.up1(x6)
        x4_cropped = self.crop_and_copy(x4, up1)
        concat1 = torch.cat([up1, x4_cropped], dim=1)
        x5 = self.dec_conv0(concat1)
        x6 = self.dec_conv1(x5)
        
        up2 = self.up2(x6)
        x2_cropped = self.crop_and_copy(x2, up2)
        concat2 = torch.cat([up2, x2_cropped], dim=1)
        x7 = self.dec_conv2(concat2)
        x8 = self.dec_conv3(x7)
        
        # Final output
        out = self.final(x8)
        return self.sigmoid(out)
    
    def crop_and_copy(self, enc_features, dec_features):
        _, _, D_enc, H_enc, W_enc = enc_features.size()
        _, _, D_dec, H_dec, W_dec = dec_features.size()
        
        # Calculate the starting indices for cropping
        start_D = (D_enc - D_dec) // 2
        start_H = (H_enc - H_dec) // 2
        start_W = (W_enc - W_dec) // 2
        
        # Crop the encoder features to the size of the decoder features from the center
        cropped_enc_features = enc_features[:, :, start_D:start_D + D_dec, start_H:start_H + H_dec, start_W:start_W + W_dec]
        return cropped_enc_features

    def to(self, device):
        super().to(device)
        self.device = device
