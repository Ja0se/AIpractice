#픽셀이 512 512고정일때
import cv2
import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from _1723871_vdsr_loader import TrainDataset, TestDataset
class RetBox(nn.Module):
    def __init__(self,in_channels=200,widths=8,heights=8):
        super(RetBox,self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels * widths * heights, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=in_channels*widths*heights)
        self.relu=nn.ReLU()
        self.dropOut2d=nn.Dropout2d()

    def forward(self,x):
        out = self.relu(self.fc1(x))
        out=self.dropOut2d(out)
        out=self.relu(self.fc2(out))

        out=self.relu(self.fc2(out))
        out=self.relu(self.fc3(out))
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.maxPool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=5,padding=2)
        self.conv1_=nn.Conv2d(in_channels=64,out_channels=100,kernel_size=5,padding=2)
        self.conv2=nn.Conv2d(in_channels=100,out_channels=130,kernel_size=5,padding=2)
        self.conv3=nn.Conv2d(in_channels=130,out_channels=150,kernel_size=3,padding=1)
        self.conv3_=nn.Conv2d(in_channels=150,out_channels=170,kernel_size=3,padding=1)
        self.conv3_1=nn.Conv2d(in_channels=170,out_channels=180,kernel_size=3,padding=1)
        self.conv3_2=nn.Conv2d(in_channels=180,out_channels=200,kernel_size=3,padding=1)



        self.conv4=nn.Conv2d(in_channels=200,out_channels=180,kernel_size=3,padding=1)
        self.conv5=nn.Conv2d(in_channels=180,out_channels=170,kernel_size=5,padding=2)
        self.conv6 = nn.Conv2d(in_channels=170, out_channels=150, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels=150, out_channels=130, kernel_size=5, padding=2)
        self.conv8 = nn.Conv2d(in_channels=130, out_channels=100, kernel_size=5, padding=2)
        self.conv9 = nn.Conv2d(in_channels=100, out_channels=64, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, padding=2)
        self.upSample=nn.Upsample(scale_factor=2)
        self.fc1 = nn.Linear(in_features=200*8*8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=200*8*8)

        self.dropOut2d=nn.Dropout2d()
    def forward(self, x):
        #print('x: ',x.shape)
        out=self.relu(self.conv1(x))
        out=self.maxPool2d(out)
        out=self.relu(self.conv1_(out))
        out=self.maxPool2d(out)
        out=self.relu(self.conv2(out))
        out=self.maxPool2d(out)
        out=self.relu(self.conv3(out))
        out = self.maxPool2d(out)
        out = self.relu(self.conv3_(out))
        out = self.maxPool2d(out)
        out = self.relu(self.conv3_1(out))
        out = self.maxPool2d(out)
        out = self.relu(self.conv3_2(out))
        #print('conv3 out: ',out.shape)

        out=out.view(-1,200*8*8)
        #print('out view : ',out.shape)
        out = self.relu(self.fc1(out))
        out = self.dropOut2d(out)
        out = self.relu(self.fc2(out))

        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        #print('linear : ',out.shape)
        out=out.view(-1,200,8,8)
        #print('out view : ',out.shape)
        out = self.relu(self.conv4(out))
        out = self.upSample(out)
        out = self.relu(self.conv5(out))
        out = self.upSample(out)
        out = self.relu(self.conv6(out))
        out = self.upSample(out)
        out = self.relu(self.conv7(out))
        out = self.upSample(out)
        out = self.relu(self.conv8(out))
        out = self.upSample(out)
        out = self.relu(self.conv9(out))
        out = self.upSample(out)
        out=self.conv10(out)
        #print('out',out.shape)
        return out
# checking img trans
# model=CNN()
# test_dataset=TestDataset()
# test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)
# for data in test_dataloader:
#     inputImg,labelImg,imgName=data
#     print('next img-----------------')
#     model(inputImg)

model=CNN()
batch_size=16
learning_rate=5e-4 #0.0005
training_epochs=1
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