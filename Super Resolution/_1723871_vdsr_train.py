import cv2
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from _1723871_vdsr_loader import TrainDataset, TestDataset

class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=9,padding=4)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=7,padding=3)
        self.conv3=nn.Conv2d(in_channels=32,out_channels=16,kernel_size=1,padding=0)
        self.conv4=nn.Conv2d(in_channels=16,out_channels=1,kernel_size=5,padding=2)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        out=self.relu(self.conv3(out))
        out=self.conv4(out)
        out=torch.add(out,x)
        return out

model=ARCNN()
batch_size=16
learning_rate=5e-4 #0.0005
training_epochs=15
loss_function=nn.MSELoss()

optimizer = optim.Adam([
    {'params':model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters()},
    {'params': model.conv4.parameters(), 'lr' : learning_rate * 10}
], lr =learning_rate)

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
torch.save(model.state_dict(),"vdsrarcnn.pth")
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