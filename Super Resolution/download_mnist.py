import torchvision.datasets as dataset
import torchvision.transforms as transform
import matplotlib.pyplot as plt

mnist_train=dataset.MNIST(root='.\\dataset\\',
                          train=True,
                          transform=transform.ToTensor(),
                          download=True
                          )

mnist_test=dataset.MNIST(root='\\dataset\\',
                         train=False,
                         transform=transform.ToTensor(),
                         download=True
                         )

print(len(mnist_train))
first_dataset=mnist_train[0]

print(first_dataset[0].shape)
print(first_dataset[1])
plt.imshow(first_dataset[0][0,:,:],cmap='gray')
plt.show()