import cv2
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from _1723871_vdsr_loader import TrainDataset, TestDataset

class DenseBlock(nn.Module):
    def __init__(self,n_channels=64,kernel_size=3):
        super(DenseBlock,self).__init__()
        padding=kernel_size//2

        self.conv1=nn.Conv2d(n_channels,n_channels,kernel_size,padding=padding)
        self.conv2=nn.Conv2d(n_channels*2,n_channels,kernel_size,padding=padding)
        self.conv3=nn.Conv2d(n_channels*3,n_channels,kernel_size,padding=padding)
        self.relu=nn.ReLU()
    def forward(self,x):
        #print('shape of x:\t',x.shape)
        out1=self.relu(self.conv1(x))
        out1=torch.cat([x,out1],dim=1)
        #print('shape of out1:\t',out1.shape)
        out2=self.relu(self.conv2(out1))
        out2=torch.cat([out1,out2],dim=1)
        #print('shape of out2:\t',out2.shape)
        out3=self.relu(self.conv3(out2))
        #print('shape of out3:\t',out3.shape)
        return out3

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.inputConv=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1)
        middleConvs = []
        for i in range(3):
            #middleConvs.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
            #middleConvs.append(nn.ReLU())
            middleConvs.append(DenseBlock(64,3))
        self.middleConvs = nn.Sequential(*middleConvs)

        self.outputConv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.inputConv(x))
        out = self.middleConvs(out)
        out = self.outputConv(out)
        out = torch.add(out, x)
        return out


model=CNN()
batch_size=16
learning_rate=1e-4 #0.0005
training_epochs=15
loss_function=nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=learning_rate)

train_dataset=TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

test_dataset=TestDataset()
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)



for epoch in range(training_epochs):
    avg_cost=0
    total_batch=len(train_dataloader)

    for data in train_dataloader:
        inputImg,labelImg=data

        predImg=model(inputImg)

        loss=loss_function(predImg,labelImg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_cost+=loss/total_batch
    print('Epoch: {} Loss = {}'.format(epoch+1,avg_cost))
torch.save(model.state_dict(),"testcnn.pth")
print('Learning finished')

def calc_psnr(orig,pred):
    return 10. * torch.log10(1./torch.mean((orig-pred)**2))

#각 이미지별 psnr 저장을 위한 리스트 생성
jpeg_PSNRs=[]
arcnn_PSNRs=[]

for data in test_dataloader:
    inputImg,labelImg,imgName=data

    with torch.no_grad():
        predImg=model(inputImg).clamp(0.0,1.0)
        #한개 이미지에 대한 bicubic,srcnn 각각  psnr저장
    jpeg_PSNRs.append(calc_psnr(labelImg,inputImg))
    arcnn_PSNRs.append(calc_psnr(labelImg,predImg))
    #예측 이미지 저장
    predImg=np.array(predImg * 255, dtype=np.uint8)
    predImg=np.transpose(predImg[0,:,:,:],[1,2,0])
    name=imgName[0].split('\\')

    cv2.imwrite('.\\AR_dataset\\Set5_vdsr/'+name[-1],predImg)

print('Average PSNR (jpeg)\t : %.4fdB'%(sum(jpeg_PSNRs)/len(jpeg_PSNRs)))
print('Average PSNR (arcnn)\t : %.4fdB'%(sum(arcnn_PSNRs)/len(arcnn_PSNRs)))