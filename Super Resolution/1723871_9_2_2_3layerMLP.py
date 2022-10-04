import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

#Dataset 선언
mnist_train=dataset.MNIST(root='.\\dataset\\',train=True,transform=transform.ToTensor(),download=True)
mnist_test=dataset.MNIST(root='\\dataset\\',train=False,transform=transform.ToTensor(),download=True)

#MLP structure 선언
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features=784,out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)
        self.ReLU=nn.ReLU()
    def forward(self,x):
        out=self.ReLU(self.fc1(x))
        out=self.ReLU(self.fc2(out))
        out=self.fc3(out)
        return out

#Hyper-parameter 선언
batch_size=100
learning_rate=0.1
training_epochs=15 #반복학습 횟수
loss_function=nn.CrossEntropyLoss()
network=MLP()
optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)

data_loader=DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True,drop_last=True)

#MLP학습을 위한 반복문 선언
for epoch in range(training_epochs):#Training loop(Epoch)
    avg_cost=0
    total_batch=len(data_loader)

    for img,label in data_loader: # Training loop(Iteration)
        img=img.view(-1,28*28) #이미지 평탄화

        pred = network(img) # MLP를 이용해 손글씨 예측

        loss=loss_function(pred,label)#Loss function 을 이용해 loss 계산
        optimizer.zero_grad()
        loss.backward()#loss에 대한 기울기 계산
        optimizer.step()#weight update
        avg_cost+=loss/total_batch
    print('Epoch: {} Loss = {}'.format(epoch+1,avg_cost))
torch.save(network.state_dict(),"basic_MLP.pth")#학습이 완료된 파라미터 저장
print('Learning finished')

with torch.no_grad(): #기울기 계산을 제외하고 계산
    img_test=mnist_test.data.view(-1,28*28).float()
    label_test=mnist_test.targets

    prediction=network(img_test)
    correct_prediction=torch.argmax(prediction,1)==label_test #예측 값이 가장 높은 숫자와 정답데이터가 일치하는지 확인
    accuracy=correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())