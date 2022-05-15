import numpy as np
from ImgProcessing import ImgProcessing

class MF():
    def __init__(self, ratings, K, alpha, beta, iteration, verbose=True):
        self.R = np.array(ratings)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iteration = iteration
        self.verbose = verbose

    def MSE(self):
        xs, ys=self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction=self.b+self.b_u[x]+self.b_d[y]+self.P[x, :].dot(self.Q[y, :].T)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y]-prediction)
        self.predictions = np.array(self.errors)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    def train(self):
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        row, col=self.R.nonzero()
        self.samples = [(i, j, self.R[i, j])for i, j in zip(row, col)]
        training_process=[]
        for i in range(self.iteration):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.MSE()
            training_process.append((i+1, mse))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration: %d ; error = %.4f' % (i+1, mse))
        return training_process

    def sgd(self):
        for i, j, k in self.samples:
            prediction=self.b+self.b_u[i]+self.b_d[j]+self.P[i, :].dot(self.Q[j, :].T)
            e=(k-prediction)

            self.b_u[i]+=self.alpha*(e-self.beta*self.b_u[i])
            self.b_d[j]+=self.alpha*(e-self.beta*self.b_d[j])

            self.P[i, :]+=self.alpha*(e*self.Q[j, :]-self.beta*self.P[i, :])
            self.Q[j, :]+=self.alpha*(e*self.P[i, :]-self.beta*self.Q[j, :])

    def get_Matrix(self):
        return self.b+self.b_u[:,np.newaxis]+self.b_d[np.newaxis:,]+self.P.dot(self.Q.T)