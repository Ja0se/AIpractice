import torch
from torch import nn
from torchsummary import summary

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR,self).__init__()
        self.inputConv=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        middleConvs=[]
        for i in range(18):
            middleConvs.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1))
            middleConvs.append(nn.ReLU())
        self.middleConvs=nn.Sequential(*middleConvs)

        self.outputConv=nn.Conv2d(in_channels=64,out_channels=1,kernel_size=3,padding=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.relu(self.inputConv(x))
        out=self.middleConvs(out)
        out=self.outputConv(out)
        out=torch.add(out,x)
        return out

network=VDSR()
summary(network,(1,224,224),device='cpu')