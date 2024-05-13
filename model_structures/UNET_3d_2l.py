import torch
import torch.nn as nn
from torch import device, cuda

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        
        # Encoder
        self.enc_conv0 = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16))
        self.enc_conv1 = nn.Sequential(nn.Conv3d(16, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16))
        self.pool1 = nn.MaxPool3d(2)
        self.enc_conv2 = nn.Sequential(nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32))
        self.enc_conv3 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32))
        self.pool2 = nn.MaxPool3d(2)
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        self.dec_conv1 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(32))
        self.up2 = nn.ConvTranspose3d(32, 16, 2, stride=2)
        self.dec_conv2 = nn.Sequential(nn.Conv3d(32, 16, 3, padding=1), nn.ReLU(), nn.BatchNorm3d(16))
        
        # Final layer
        self.final = nn.Conv3d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv0(x)
        x2 = self.enc_conv1(x1)
        p1 = self.pool1(x2)
        
        x3 = self.enc_conv2(p1)
        x4 = self.enc_conv3(x3)
        p2 = self.pool2(x4)
        
        # Decoder
        up1 = self.up1(p2)
        x4_cropped = self.crop_and_copy(x4, up1)
        concat1 = torch.cat([up1, x4_cropped], dim=1)
        x5 = self.dec_conv1(concat1)
        
        up2 = self.up2(x5)
        x2_cropped = self.crop_and_copy(x2, up2)
        concat2 = torch.cat([up2, x2_cropped], dim=1)
        x6 = self.dec_conv2(concat2)
        
        # Final output
        out = self.final(x6)
        return self.sigmoid(out)
    
    def crop_and_copy(self, enc_features, dec_features):
        _, _, D, H, W = dec_features.size()
        enc_features = enc_features[:, :, :D, :H, :W]
        return enc_features

    def to(self, device):
        super().to(device)
        self.device = device
