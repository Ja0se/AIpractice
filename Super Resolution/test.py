import cv2
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from sr_dataloader import TrainDataset, TestDataset
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self,in_channels=64,out_channels=64,kernel_size=3):
        super(ResBlock,self).__init__()

        padding=kernel_size//2

        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size,padding=padding)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size,padding=padding)
        self.relu=nn.ReLU()
    def forward(self,x):
        out=self.relu(self.conv1(x))
        out=self.conv2(out)
        out=torch.add(out,x)
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        cfgs = [(3, 64, 64, 1),
                (1, 64, 64, 1),
                ]
        cfgs2=[(5, 64, 64, 1),
                (1, 64, 64, 1),
                (2, 64, 64, 1),
                ]
        self.inputConv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        resBlocks = []
        for cfg in cfgs:
            n_blocks, in_channels, out_channels, stride = cfg
            for idx in range(n_blocks):
                resBlocks.append(ResBlock(in_channels, out_channels, 3))
        self.resBlocks = nn.Sequential(*resBlocks)
        resBlocks2=[]
        for cfg in cfgs2:
            n_blocks, in_channels, out_channels, stride = cfg

            for idx in range(n_blocks):
                resBlocks2.append(ResBlock(in_channels, out_channels, 3))
        self.resBlocks2 = nn.Sequential(*resBlocks2)
        self.conv1=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=64*2,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64*3,out_channels=64,kernel_size=3,padding=1)
        self.relu = nn.ReLU()
        self.outputConv1=nn.Conv2d(in_channels=64,out_channels=3,kernel_size=7,padding=3)
    def forward(self, x):
        #input
        out = self.relu(self.inputConv(x))
        #res
        out = self.resBlocks(out)
        #dense
        out1=self.relu(self.conv1(out))
        out1=torch.cat([out,out1],dim=1)
        out2=self.relu(self.conv2(out1))
        out2=torch.cat([out1,out2],dim=1)
        out3=self.relu(self.conv3(out2))
        #skip
        out3=torch.add(out3,out)
        #res
        out4=self.resBlocks2(out3)
        #output
        out4=self.outputConv1(out4)
        #skip
        out4=torch.add(out4,x)
        return out4
USE_CUDA = torch.cuda.is_available()
print(USE_CUDA)
device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)
model=CNN().to(device)
model.load_state_dict(torch.load('resblockcnn.pth'))

def calc_psnr(orig,pred):
    return 10. * torch.log10(1./torch.mean((orig-pred)**2))
test_dataset=TestDataset()
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)


#각 이미지별 psnr 저장을 위한 리스트 생성
jpeg_PSNRs=[]
arcnn_PSNRs=[]
for data in test_dataloader:
    inputImg,labelImg,imgName=data
    inputImg=inputImg.to(device)
    labelImg=labelImg.to(device)
    with torch.no_grad():
        predImg=model(inputImg).clamp(0.0,1.0)
        #한개 이미지에 대한 bicubic,srcnn 각각  psnr저장
    jpeg_PSNRs.append(calc_psnr(labelImg,inputImg))
    arcnn_PSNRs.append(calc_psnr(labelImg,predImg))
    #예측 이미지 저장
    predImg=np.array(predImg.cpu() * 255, dtype=np.uint8)
    predImg=np.transpose(predImg[0,:,:,:],[1,2,0])
    name=imgName[0].split('\\')

    print(name[-1])
    print('bic\t',jpeg_PSNRs[-1])
    print('ar\t',arcnn_PSNRs[-1])
    cv2.imwrite('.\\SR_dataset\\Set5_vdsr/'+name[-1],predImg)

print('Average PSNR (jpeg)\t : %.4fdB'%(sum(jpeg_PSNRs)/len(jpeg_PSNRs)))
print('Average PSNR (arcnn)\t : %.4fdB'%(sum(arcnn_PSNRs)/len(arcnn_PSNRs)))