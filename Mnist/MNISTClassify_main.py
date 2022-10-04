import tensorflow as tf
from ImageClassifier import ImageClassifier
import matplotlib.pyplot as plt
import glob
from PIL import Image
import numpy as np
def MNIST_Classify():
    mnist_train_data, mnist_test_data = tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #### use x_train (Images(Feature vectors)), y_train (Class ground truths) as training set
    x_train, y_train = mnist_train_data
    #### use x_test (Images(Feature vectors)), y_test (Class ground truths) as test set
    x_test, y_test = mnist_test_data
    # use x_test my image
    
    ############ Write your codes here - begin
    #0~1 change
    x_train,x_test=x_train/255.,x_test/255.
    #,(my_test/255.0-1)*-1
    #make model and train
    model=ImageClassifier(28,28,10)
    model.configure_model()
    print('train start')
    model.fit(x_train,y_train,10)
    #result
    predicted_labels=model.predict(x_test)
    predicted_labels=tf.math.argmax(predicted_labels,axis=1)
    #print
    plt.figure(figsize=(10,10))
    accuracy=0.
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i],'gray')
        plt.xlabel(class_names[predicted_labels[i]])
        if predicted_labels[i]==y_test[i]:
            accuracy+=1.
    print('accuracy : %.4f'%accuracy)
    plt.show()
    
    ############ Write your codes here - end

def MNIST_My():
    mnist_train_data, mnist_test_data = tf.keras.datasets.mnist.load_data()
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    #### use x_train (Images(Feature vectors)), y_train (Class ground truths) as training set
    x_train, y_train = mnist_train_data
   
    my_test=[]
    for image_file in glob.glob('./Image/dataset/*.png'):
        #open file and convert 256 black and white
        img=Image.open(image_file).convert("L")
        print(img.size)
        img=np.array(img)
        img=(img/255.0-1)*-1
        my_test.append(img)
    my_test=np.array(my_test)
    my_ytest=['0','1','2','3','4','5','6','7','8','9']
    model=ImageClassifier(28,28,10)
    model.configure_model()
    print('train start')
    model.fit(x_train,y_train,10)
    #result
    predicted_labels=model.predict(my_test)
    predicted_labels=tf.math.argmax(predicted_labels,axis=1)
    #print
    plt.figure(figsize=(10,10))
    accuracy=0.
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(my_test[i],'gray')
        plt.xlabel(class_names[predicted_labels[i]])
        if predicted_labels[i]==my_ytest[i]:
            accuracy+=1.
    print('accuracy : %.4f'%accuracy)
    plt.show()
    
if __name__ == '__main__':
    MNIST_Classify()
    MNIST_My()
