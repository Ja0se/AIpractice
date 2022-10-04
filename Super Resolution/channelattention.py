import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self,n_channels=64):
        super(ChannelAttention,self).__init__()

        self.adaptiveAvgPool2d=nn.AdaptiveAvgPool2d((1,1))
        self.conv1=nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=n_channels,out_channels=n_channels,kernel_size=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        print('shape of x: ',x.shape)
        out=self.adaptiveAvgPool2d(x)
        out=self.conv1(out)
        out=self.conv2(out)
        out=self.sigmoid(out)
        print('shape of out: ',out.shape)
        CA_map=out.expand_as(x)
        print('shape of CA_map: ',CA_map.shape)
        out=x*CA_map
        print('shape of out: ',out.shape)
        return out

network=ChannelAttention()
test=torch.zeros([1,64,224,224],dtype=torch.float32)
network(test)