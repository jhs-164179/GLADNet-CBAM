from torch import nn
from torch.nn import functional as F


# Detail Reconstruction
class DR(nn.Module):
    def __init__(self):
        super(DR, self).__init__()
        self.a_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=67, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.a_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.a_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.a_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.a_conv5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.a_conv1(x)
        x = self.a_conv2(x)
        x = self.a_conv3(x)
        x = self.a_conv4(x)
        x = self.a_conv5(x)
        return x
