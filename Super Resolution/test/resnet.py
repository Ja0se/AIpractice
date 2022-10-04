import cv2
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from _1723871_vdsr_loader import TrainDataset, TestDataset

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
        #print('error first: ', out.shape)
        out=self.conv2(out)
        #print('error second: ', out.shape)
        if self.stride!=1:
            x=self.maxPool2d(x)
            x=torch.cat([x,x],dim=1)
        #print('error : ',out.shape)
        #print('error x : ',x.shape)
        if out.size==x.size:out=torch.add(out,x)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        cfgs = [(3, 64, 64, 1),
                (1, 64, 128, 2),
                (3, 128, 128, 1),
                (1, 128, 256, 2),
                (5, 256, 256, 1),
                (1, 256, 512, 2),
                (2, 512, 512, 1),
                ]
        self.inputConv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        resBlocks = []
        for cfg in cfgs:
            n_blocks, in_channels, out_channels, stride = cfg

            for idx in range(n_blocks):
                resBlocks.append(ResBlock(in_channels, out_channels, 3, stride))
        self.resBlocks = nn.Sequential(*resBlocks)

        self.fc = nn.Linear(in_features=512, out_features=1000)
        self.fc1 = nn.Linear(in_features=1000,out_features=1000)
        self.fc2 = nn.Linear(in_features=1000,out_features=512)
        self.dropOut2d = nn.Dropout2d()
        self.upSample=nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()
        self.maxPool2d = nn.MaxPool2d(2, stride=2)
        self.adaptAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.outputConv1=nn.Conv2d(in_channels=512,out_channels=1,kernel_size=7,padding=3)
    def forward(self, x):
        out = self.relu(self.inputConv(x))
       # print('inputConv : ',out.shape)
        out = self.maxPool2d(out)
       # print('maxpool : ',out.shape)
        out = self.resBlocks(out)
        #print('resBlocks : ',out.shape)
        out = self.adaptAvgPool2d(out)
        #print('adaptAvgpool : ',out.shape)
        out = out.view(-1, 512)
        out = self.relu(self.fc(out))
        out=self.dropOut2d(out)
        out=self.relu(self.fc1(out))

        out=self.relu(self.fc1(out))
        out=self.relu(self.fc2(out))
        out=out.view(-1,512,1,1)
        out=self.upSample(out)
        out=self.outputConv1(out)
        print('out : ',out.shape)

        return out

# # checking img trans
model=CNN()
test_dataset=TestDataset()
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)
for data in test_dataloader:
    inputImg,labelImg,imgName=data
    print('next img---------------------------------------------')
    print(inputImg.shape)
    model(inputImg)



# model=CNN()
# batch_size=16
# learning_rate=5e-4 #0.0005
# training_epochs=1
# loss_function=nn.MSELoss()
#
# optimizer = optim.Adam(model.parameters(),lr=learning_rate)
#
# train_dataset=TrainDataset()
# train_dataloader = DataLoader(dataset=train_dataset,
#                               batch_size=batch_size,
#                               shuffle=True,
#                               drop_last=True)
#
# test_dataset=TestDataset()
# test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)
#
#
#
# for epoch in range(training_epochs):
#     avg_cost=0
#     total_batch=len(train_dataloader)
#
#     for data in train_dataloader:
#         inputImg,labelImg=data
#
#         predImg=model(inputImg)
#
#         loss=loss_function(predImg,labelImg)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         avg_cost+=loss/total_batch
#     print('Epoch: {} Loss = {}'.format(epoch+1,avg_cost))
# torch.save(model.state_dict(),"vdsrarcnn.pth")
# print('Learning finished')
#
# def calc_psnr(orig,pred):
#     return 10. * torch.log10(1./torch.mean((orig-pred)**2))
#
# #각 이미지별 psnr 저장을 위한 리스트 생성
# jpeg_PSNRs=[]
# arcnn_PSNRs=[]
#
# for data in test_dataloader:
#     inputImg,labelImg,imgName=data
#
#     with torch.no_grad():
#         predImg=model(inputImg).clamp(0.0,1.0)
#         #한개 이미지에 대한 bicubic,srcnn 각각  psnr저장
#     jpeg_PSNRs.append(calc_psnr(labelImg,inputImg))
#     arcnn_PSNRs.append(calc_psnr(labelImg,predImg))
#     #예측 이미지 저장
#     predImg=np.array(predImg * 255, dtype=np.uint8)
#     predImg=np.transpose(predImg[0,:,:,:],[1,2,0])
#     name=imgName[0].split('\\')
#
#     cv2.imwrite('.\\AR_dataset\\Set5_vdsr/'+name[-1],predImg)
#
# print('Average PSNR (jpeg)\t : %.4fdB'%(sum(jpeg_PSNRs)/len(jpeg_PSNRs)))
# print('Average PSNR (arcnn)\t : %.4fdB'%(sum(arcnn_PSNRs)/len(arcnn_PSNRs)))