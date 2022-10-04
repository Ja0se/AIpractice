class sigmoid_regressor:
  a,b,fit_epoch=0.,0.,0
  def __intit__(self,a,b,fit_epoch):
    self.a=a
    self.b=b
    self.fit_epoch=fit_epoch

  def sigmoid(self,x):
    return 1 / (1 + np.exp(-x))

  def fit(self,epochs,learing_rate,x,y):
    for i in range(1,epochs+1):
      for x_data,y_data in zip(x,y):
        z=self.sigmoid(self.a*x_data+self.b)-y_data
        a_diff=x_data*z
        b_diff=z
        self.a=self.a-learning_rate*a_diff
        self.b=self.b-learning_rate*b_diff
      if i%10000==0:
        print('epoch=%.4f, 기울기 a= %.2f, y절편 b = %.2f'%(self.fit_epoch+i,self.a,self.b))
    self.fit_epoch+=epochs

  def pred(self):
    return self.a,self.b

  def Epoch(self):
    return self.fit_epoch
  
if __name__=='__main__':
  curve=sigmoid_regressor()
  learning_rate=0.01
  epochs=10000
  curve.fit(epochs,learning_rate,pred_idx,Y_pred)
  a,b=curve.pred()
