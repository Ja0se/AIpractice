import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import tensorflow as tf
from MLP import MLP
import numpy as np
from sklearn.linear_model import SGDClassifier

def CircleClassify():
    # generating data
    n_samples = 400
    noise = 0.02
    factor = 0.5
    #### use x_train (Feature vectors), y_train (Class ground truths) as training set
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #### use x_test (Feature vectors) as test set
    #### you do not use y_test for this assignment.
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #print(y_test)
    #### visualizing training data distribution
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
    plt.title("Train data distribution")
    plt.show()
    
    ############ Write your codes here - begin
    batch_size=1
    epochs=300
    #slp
    slp_classifier = MLP(hidden_layer_conf=None, num_output_nodes=2)
    slp_classifier.build_model()
    slp_classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
    
    #mlp
    mlp_classifier = MLP(hidden_layer_conf=[3,3], num_output_nodes=2)
    mlp_classifier.build_model()
    mlp_classifier.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)
    
    ### test
    slp_prediction = slp_classifier.predict(x=x_test, batch_size=batch_size)
    mlp_prediction = mlp_classifier.predict(x=x_test, batch_size=batch_size)
    
    plt.scatter(x_test[:,0], x_test[:,1], c=slp_prediction, marker='.')
    plt.title("Train SLP data result")
    plt.show()
    
    plt.scatter(x_test[:,0], x_test[:,1], c=mlp_prediction, marker='.')
    plt.title("Train MLP data result")
    plt.show()

    ############ Write your codes here - end
    
def SGDClassify():
    n_samples = 400
    noise = 0.02
    factor = 0.5
    #### use x_train (Feature vectors), y_train (Class ground truths) as training set
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    #### use x_test (Feature vectors) as test set
    #### you do not use y_test for this assignment.
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    model=SGDClassifier(max_iter=300)
    model.fit(x_train,y_train)
    pred_classified=model.predict(x_test)
    
    a = model.coef_[0,0]
    b = model.coef_[0,1]
    c = model.intercept_
    
    x=np.linspace(-1.5,1.5,100)
    y=(-a/b)*x-(c/b)
    plt.plot(x,y)
    plt.scatter(x_test[:,0],x_test[:,1],c=pred_classified,marker='.')
    plt.title('SGDClassifier data result')
    plt.show()
    
if __name__ == '__main__':
    #CircleClassify()
    SGDClassify()