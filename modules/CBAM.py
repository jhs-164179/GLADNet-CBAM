import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, input_dim, output_dim, r=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim // r),
            nn.ReLU(),
            nn.Linear(output_dim // r, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = avg_out + max_out
        x = F.sigmoid(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, (7, 7), 1, 1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = F.sigmoid(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(in_channels, out_channels)
        self.SpatialAttention = SpatialAttention(in_channels)

    def forward(self, x):
        x *= self.ChannelAttention(x)
        x *= self.SpatialAttention(x)
        return x
