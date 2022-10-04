import torch
from torch import nn
from torchsummary import summary
class ResBlock(nn.Module):
    def __init__(self,in_channels=64,out_channels=64,kernel_size=3,stride=1):
        super(ResBlock,self).__init__()
        self.stride=stride
        padding=kernel_size//2

        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size,padding=padding)
        self.relu=nn.ReLU()
        self.maxPool2d=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        out=self.relu(self.conv1(x))
        out=self.conv2(out)

        if self.stride!=1:
            x=self.maxPool2d(x)
            x=torch.cat([x,x],dim=1)
        out=torch.add(out,x)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34,self).__init__()
        cfgs=[(3,64,64,1),
              (1,64,128,2),
              (3,128,128,1),
              (1,128,256,2),
              (5,256,256,1),
              (1,256,512,2),
              (2,512,512,1),
              ]
        self.inputConv=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        resBlocks=[]
        for cfg in cfgs:
            n_blocks,in_channels,out_channels,stride=cfg

            for idx in range(n_blocks):
                resBlocks.append(ResBlock(in_channels,out_channels,3,stride))
        self.resBlocks=nn.Sequential(*resBlocks)

        self.fc=nn.Linear(in_features=512,out_features=1000)

        self.relu=nn.ReLU()
        self.maxPool2d=nn.MaxPool2d(2,stride=2)
        self.adaptAvgPool2d=nn.AdaptiveAvgPool2d((1,1))

    def forward(self,x):
        out=self.relu(self.inputConv(x))
        out=self.maxPool2d(out)
        out=self.resBlocks(out)
        out=self.adaptAvgPool2d(out)
        out=out.view(-1,512)
        out=self.fc(out)
        return out

network=ResNet34()
summary(network,(3,224,224),device='cpu')