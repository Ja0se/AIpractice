import torch
from torch import nn

class DenseBlock(nn.Module):
    def __init__(self,n_channels=64,kernel_size=3):
        super(DenseBlock,self).__init__()
        padding=kernel_size//2

        self.conv1=nn.Conv2d(n_channels,n_channels,kernel_size,padding=padding)
        self.conv2=nn.Conv2d(n_channels*2,n_channels,kernel_size,padding=padding)
        self.conv3=nn.Conv2d(n_channels*3,n_channels,kernel_size,padding=padding)
        self.relu=nn.ReLU()
    def forward(self,x):
        print('shape of x:\t',x.shape)
        out1=self.relu(self.conv1(x))
        out1=torch.cat([x,out1],dim=1)
        print('shape of out1:\t',out1.shape)
        out2=self.relu(self.conv2(out1))
        out2=torch.cat([out1,out2],dim=1)
        print('shape of out2:\t',out2.shape)
        out3=self.relu(self.conv3(out2))
        print('shape of out3:\t',out3.shape)
        return out3

network=DenseBlock()
test=torch.zeros([1,64,224,224],dtype=torch.float32)
network.forward(test)