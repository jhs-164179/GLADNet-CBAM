from torch import nn
from torch.nn import functional as F


# Illumination Distribution Estimation
class IDE(nn.Module):
    def __init__(self):
        super(IDE, self).__init__()
        self.p_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.p_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.p_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.p_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.p_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.p_conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.p_deconv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.p_deconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.p_deconv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.p_deconv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.p_deconv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.p_deconv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 48
        p_conv1 = self.p_conv1(x)
        # 24
        p_conv2 = self.p_conv2(p_conv1)
        # 12
        p_conv3 = self.p_conv3(p_conv2)
        # 6
        p_conv4 = self.p_conv4(p_conv3)
        # 3
        p_conv5 = self.p_conv5(p_conv4)
        # 1
        p_conv6 = self.p_conv6(p_conv5)
        # 3
        p_deconv1 = F.interpolate(p_conv6, size=(3, 3), mode='nearest')
        p_deconv1 = self.p_deconv1(p_deconv1)
        p_deconv1 = p_deconv1 + p_conv5
        # 6
        p_deconv2 = F.interpolate(p_deconv1, size=(6, 6), mode='nearest')
        p_deconv2 = self.p_deconv2(p_deconv2)
        p_deconv2 = p_deconv2 + p_conv4
        # 12
        p_deconv3 = F.interpolate(p_deconv2, size=(12, 12), mode='nearest')
        p_deconv3 = self.p_deconv3(p_deconv3)
        p_deconv3 = p_deconv3 + p_conv3
        # 24
        p_deconv4 = F.interpolate(p_deconv3, size=(24, 24), mode='nearest')
        p_deconv4 = self.p_deconv4(p_deconv4)
        p_deconv4 = p_deconv4 + p_conv2
        # 48
        p_deconv5 = F.interpolate(p_deconv4, size=(48, 48), mode='nearest')
        p_deconv5 = self.p_deconv5(p_deconv5)
        p_deconv5 = p_deconv5 + p_conv1
        # 96
        p_deconv6 = F.interpolate(p_deconv5, size=(96, 96), mode='nearest')
        p_deconv6 = self.p_deconv6(p_deconv6)

        return p_deconv6
