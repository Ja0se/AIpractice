import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#Dataset 선언
mnist_train=dataset.MNIST(root='.\\dataset\\',train=True,transform=transform.ToTensor(),download=True)
mnist_test=dataset.MNIST(root='\\dataset\\',train=False,transform=transform.ToTensor(),download=True)

#MLP structure 선언  -> CNN 구조 정의
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1=nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride= 1, padding=0)
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride= 1, padding=0)
        self.fc1=nn.Linear(in_features=256,out_features=120)
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
        out=out.view(-1,256)#feature map 평탄화
        out=self.relu(self.fc1(out))
        out=self.relu(self.fc2(out))
        out=self.fc3(out)
        return out

network=LeNet5()
network.load_state_dict(torch.load('leNet5_mnist.pth'))
#MLP성능 측정 -> 이미지 전처리(평탄화) 부분 제거

with torch.no_grad(): #기울기 계산을 제외하고 계산
    img_test=mnist_test.data.unsqueeze(1).float()#평탄화 제거, 새로운 차원(channel) 추가 10000*28*28  -> 10000*1*28*28
    label_test=mnist_test.targets

    prediction=network(img_test)
    correct_prediction=torch.argmax(prediction,1)==label_test #예측 값이 가장 높은 숫자와 정답데이터가 일치하는지 확인
    accuracy=correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
    failIdx=list(correct_prediction).index(False)

#한개의 test dataset에 대한 결과 확인
temp_img=mnist_test.data[failIdx]
temp_label=mnist_test.targets[failIdx]
prediction_num=torch.argmax(prediction[failIdx],0)

print('정답값은{}입니다.'.format(temp_label))
print('예측값은{}입니다.'.format(prediction_num))
plt.imshow(temp_img,cmap='gray')
plt.show()