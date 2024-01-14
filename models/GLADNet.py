import torch
from torch import nn
from torch.nn import functional as F

from modules.IDE import IDE
from modules.DR import DR


class GLADNet(nn.Module):
    def __init__(self):
        super(GLADNet, self).__init__()
        self.IDE = IDE()
        self.DR = DR()

    def forward(self, inp_):
        x = F.interpolate(inp_, size=(96, 96), mode='nearest')
        x = self.IDE(x)
        x = F.interpolate(x, size=(inp_.shape[-2], inp_.shape[-1]), mode='nearest')
        x = torch.cat([x, inp_], dim=1)
        x = self.DR(x)
        return x


