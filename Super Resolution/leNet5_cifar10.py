import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
#Dataset 선언
cifar10_train=dataset.CIFAR10(root='.\\dataset\\',train=True,transform=transform.ToTensor(),download=True)
cifar10_test=dataset.CIFAR10(root='\\dataset\\',train=False,transform=transform.ToTensor(),download=True)
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
#MLP structure 선언  -> CNN 구조 정의
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride= 1, padding=0)
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride= 1, padding=0)
        self.fc1=nn.Linear(in_features=400,out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=84)
        self.fc3=nn.Linear(in_features=84,out_features=10)
        #ReLU,Pooling layer 는 parameter를 가지지 않음 -> 하나만 선언해줘도 됨
        self.relu=nn.ReLU()
        self.maxPool2d=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        out=self.relu(self.conv1(x))
        out=self.maxPool2d(out)
        out=self.relu(self.conv2(out))
        out=self.maxPool2d(out)
        out=out.view(-1,400)#feature map 평탄화
        out=self.relu(self.fc1(out))
        out=self.relu(self.fc2(out))
        out=self.fc3(out)
        return out

batch_size=100
learning_rate=0.1
training_epochs=15 #반복학습 횟수
loss_function=nn.CrossEntropyLoss()
network=LeNet5()
optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)

data_loader=DataLoader(dataset=cifar10_train,batch_size=batch_size,shuffle=True,drop_last=True)

for epoch in range(training_epochs):#Training loop(Epoch)
    avg_cost=0
    total_batch=len(data_loader)

    for img,label in data_loader: # Training loop(Iteration)

        pred = network(img) # img 평탄화를 하지 않고 그대로 입력

        loss=loss_function(pred,label)#Loss function 을 이용해 loss 계산
        optimizer.zero_grad()
        loss.backward()#loss에 대한 기울기 계산
        optimizer.step()#weight update
        avg_cost+=loss/total_batch
    print('Epoch: {} Loss = {}'.format(epoch+1,avg_cost))
torch.save(network.state_dict(),"leNet5_cifar10.pth")#학습이 완료된 파라미터 저장
print('Learning finished')
#MLP성능 측정 -> 이미지 전처리(평탄화) 부분 제거

# network=LeNet5()
# network.load_state_dict(torch.load('leNet5_mnist.pth'))

with torch.no_grad(): #기울기 계산을 제외하고 계산
    img_test=torch.tensor(np.transpose(cifar10_test.data,(0,3,1,2)))/255.
    #평탄화 제거, 새로운 차원(channel) 추가 10000*28*28  -> 10000*1*28*28
    label_test=torch.tensor(cifar10_test.targets)

    prediction=network(img_test)
    correct_prediction=torch.argmax(prediction,1)==label_test #예측 값이 가장 높은 숫자와 정답데이터가 일치하는지 확인
    accuracy=correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

correct_counts=np.zeros(10)
for idx in range(cifar10_test.__len__()):
    if correct_prediction[idx]:
        correct_counts[label_test[idx]]+=1
accuracy_each_class=correct_counts/(cifar10_test.__len__()/10)
for idx in range(10):
    print('accuracy for {}\t:{}'.format(classes[idx],accuracy_each_class[idx]))