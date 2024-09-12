import sys
sys.path.append('../')
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary as model_summary
from events import Utils
class HBCO(nn.Module):
    def __init__(self, in_channels=2, out_channels=400):
        super(HBCO, self).__init__()

        self.conv2d_1 = BasicConv2d(in_channels, 64, kernel_size=1, stride=1)
        self.conv2d_2 = BasicConv2d(64, 128, kernel_size=1, stride=1)
        self.conv2d_3 = BasicConv2d(128, 256, kernel_size=1, stride=1)
        self.conv2d_4 = BasicConv2d(256, 512, kernel_size=1, stride=1)
        self.conv2d_5 = BasicConv2d(512, 1024, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1024, out_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(out_channels, out_channels)
        )


    def forward(self, x):
        #[1,1,frame,2]
        x = torch.permute(x, [0, 2, 3, 1])
        # [1,frame,2,1]
        x = self.conv2d_1(x)
        x = self.conv2d_2(x)
        x = self.conv2d_3(x)
        x = self.conv2d_4(x)
        x = self.conv2d_5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



if __name__ == "__main__":
    img = torch.ones([10, 16, 2, 1]).cuda()
    A = HBCO().cuda()
    print(A(img).shape)
    model_summary(A)


    list_flops_backbone = Utils.calculate_flops(list(A.children()))
    print(list_flops_backbone)

    size_model = 0
    for param in A.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")