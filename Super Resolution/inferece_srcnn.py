import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from sr_dataloader import TestDataset
import numpy as np
import cv2

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=9,padding=4)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,padding=2)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=5,padding=2)
        self.relu=nn.ReLU()

    def forward(self, x):
        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        out=self.conv3(out)
        return out


#Hyper-parameter 정의
model=SRCNN()
model.load_state_dict(torch.load('srcnn.pth'))


test_dataset=TestDataset()
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)

def calc_psnr(orig,pred):
    return 10. * torch.log10(1./torch.mean((orig-pred)**2))

#각 이미지별 psnr 저장을 위한 리스트 생성
bicubic_PSNRs=[]
srcnn_PSNRs=[]

for data in test_dataloader:
    inputImg,labelImg,imgName=data

    with torch.no_grad():
        predImg=model(inputImg).clamp(0.0,1.0)
        print(imgName,calc_psnr(labelImg,inputImg),calc_psnr(labelImg,predImg))
    #한개 이미지에 대한 bicubic,srcnn 각각  psnr저장
    bicubic_PSNRs.append(calc_psnr(labelImg,inputImg))
    srcnn_PSNRs.append(calc_psnr(labelImg,predImg))
    #예측 이미지 저장
    predImg=np.array(predImg * 255, dtype=np.uint8)
    predImg=np.transpose(predImg[0,:,:,:],[1,2,0])
    name=imgName[0].split('\\')

    cv2.imwrite('.\\SR_dataset\\Set5_Pred/'+name[-1],predImg)

print('Average PSNR (bicubic)\t : %.4fdB'%(sum(bicubic_PSNRs)/len(bicubic_PSNRs)))
print('Average PSNR (srcnn)\t : %.4fdB'%(sum(srcnn_PSNRs)/len(srcnn_PSNRs)))