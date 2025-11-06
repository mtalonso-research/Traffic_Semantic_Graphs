import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FrameAutoencoder(nn.Module):
    def __init__(self):
        super(FrameAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), # Input: (batch_size, 3, H, W)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Output pixel values are between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

class PretrainedAutoencoder(nn.Module):
    def __init__(self):
        super(PretrainedAutoencoder, self).__init__()
        # Load pre-trained VGG16 features as the encoder
        vgg16_features = models.vgg16(pretrained=True).features
        
        # The VGG16 features module contains pooling layers that reduce spatial dimensions.
        # For an input of 128x128, let's trace the output size.
        # VGG16 has 5 max pooling layers, each reducing spatial dimensions by half.
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        # The last convolutional layer in VGG16 features outputs 512 channels.
        # So, for 128x128 input, the encoder output will be (batch_size, 512, 4, 4)

        self.encoder = vgg16_features

        # Custom Decoder to upsample to 3x128x128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    # 64x64 -> 128x128
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)