import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#Dataset 선언
mnist_train=dataset.MNIST(root='.\\dataset\\',train=True,transform=transform.ToTensor(),download=True)
mnist_test=dataset.MNIST(root='\\dataset\\',train=False,transform=transform.ToTensor(),download=True)

#MLP structure 선언  -> 학습데이터를 불러올때는 구조가 같아야 된다.
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc=nn.Linear(in_features=784,out_features=10)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        out=self.fc(x)
        return out
#초기화 및 저장된 파라미터 파일 저장
network=MLP()
network.load_state_dict(torch.load("basic_MLP.pth"))

with torch.no_grad(): #기울기 계산을 제외하고 계산
    img_test=mnist_test.data.view(-1,28*28).float()
    label_test=mnist_test.targets

    prediction=network(img_test)
    correct_prediction=torch.argmax(prediction,1)==label_test #예측 값이 가장 높은 숫자와 정답데이터가 일치하는지 확인
    accuracy=correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

temp_data=mnist_test.data[0]
prediction=network(temp_data.view(-1,28*28).float())
print(prediction)
prediction_num=torch.argmax(prediction)

print('예측 값은 {}입니다.'.format(prediction_num))
plt.imshow(temp_data,cmap='gray')
plt.show()