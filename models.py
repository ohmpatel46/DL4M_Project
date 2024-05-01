import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import soundfile as sf
import torch.nn.functional as F

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(TemporalConvNet, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation,
                              padding=(kernel_size - 1) * dilation // 2)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class ConvTasNet(nn.Module):
    def __init__(self, num_sources, in_channels=1, encoder_channels=256, separator_channels=512,
                 num_encoder_layers=8, num_separator_layers=8, kernel_size=3):
        super(ConvTasNet, self).__init__()
        # Encoder
        encoder_layers = [TemporalConvNet(in_channels if i == 0 else encoder_channels,
                                          encoder_channels,
                                          kernel_size,
                                          stride=1,
                                          dilation=2**i)
                          for i in range(num_encoder_layers)]
        self.encoder = nn.Sequential(*encoder_layers)
        # Separator
        separator_input_channels = encoder_channels if num_encoder_layers > 0 else in_channels
        separator_layers = [TemporalConvNet(separator_input_channels if i == 0 else separator_channels,
                                            separator_channels,
                                            kernel_size,
                                            stride=1,
                                            dilation=1)
                            for i in range(num_separator_layers)]
        self.separator = nn.Sequential(*separator_layers)
        # Decoder
        decoder_layers = [TemporalConvNet(separator_channels if i == 0 else encoder_channels,
                                          encoder_channels,
                                          kernel_size,
                                          stride=1,
                                          dilation=1)
                          for i in range(num_encoder_layers)]
        self.decoder = nn.Sequential(*decoder_layers)
        # Output
        self.output = nn.Conv1d(encoder_channels, num_sources * in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoded_features = self.encoder(x)
        # Separator
        separated_features = self.separator(encoded_features)
        # Decoder
        reconstructed_features = self.decoder(separated_features)
        # Output
        separated_audio = self.output(reconstructed_features)
        return separated_audio