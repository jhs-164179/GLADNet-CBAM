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
    def __init__(self, input_dim, output_dim):
        super(CBAM, self).__init__()
        self.ChannelAttention = ChannelAttention(input_dim, output_dim)
        self.SpatialAttention = SpatialAttention(input_dim)

    def forward(self, x):
        x *= self.ChannelAttention(x)
        x *= self.SpatialAttention(x)
        return x


class IDE(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(IDE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 2, 1),
            nn.ReLU(),
            CBAM(hidden_dim, hidden_dim)
        )
        self.deconv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.deconv5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.deconv6 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(x)
        x = self.conv6(x)
        # decoder
        x = F.interpolate(x, (3, 3))(x)
        x = self.deconv1(x) + conv5
        x = F.interpolate(x, (6, 6))(x)
        x = self.deconv2(x) + conv4
        x = F.interpolate(x, (12, 12))(x)
        x = self.deconv3(x) + conv3
        x = F.interpolate(x, (24, 24))(x)
        x = self.deconv4(x) + conv2
        x = F.interpolate(x, (48, 48))(x)
        x = self.deconv5(x) + conv1
        x = F.interpolate(x, (96, 96))(x)
        x = self.deconv6(x)
        return x


class DR(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=3):
        super(DR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (3, 3), 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, (3, 3), 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class GLADNet(nn.Module):
    def __init__(self):
        super(GLADNet, self).__init__()
        self.IDE = IDE()
        self.DR = DR(input_dim=64)

    def forward(self, x):
        inp_ = x
        x = F.interpolate(inp_, (96, 96))
        x = self.IDE(x)
        x = F.interpolate(x, (inp_.shape[2], inp_.shape[3]))
        x = torch.cat([x, inp_], dim=1)
        x = self.DR(x)
        return x
