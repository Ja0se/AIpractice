import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from sr_dataloader import TrainDataset # 이전에 생성한 Training data loader 파일

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
batch_size=16
learning_rate=1e-4 # 0.0001
training_epochs=15
loss_function = nn.MSELoss()

optimizer = optim.Adam([
    {'params':model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters(), 'lr' : learning_rate * 0.1}
], lr =learning_rate)

train_dataset=TrainDataset()
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
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
torch.save(model.state_dict(),"srcnn.pth")
print('Learning finished')