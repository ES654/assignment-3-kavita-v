import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autograd import grad
from numpy.linalg import pinv
from mpl_toolkits import mplot3d

# Import Autograd modules here

def index_marks(nrows, batch_size):
    return range(1 * batch_size, int(np.ceil(nrows/batch_size)) * batch_size, batch_size)

def split(df, batch_size):
    indices = index_marks(df.shape[0], batch_size)
    return np.split(df, indices)

def mse(th, X, y):
    y_pred = np.dot(th, X.T)
    return np.sum((y - y_pred)**2/len(y))

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

        pass

    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        
        if self.fit_intercept==True:
            X = pd.DataFrame(np.hstack((np.ones((len(X),1)),X)))
        self.X = X
        self.y = y
        X_full = pd.concat([X,y], axis=1, keys=['x','y'])
        X_full = X_full.sample(frac=1, replace=False, random_state=1)
        batches = split(X_full, batch_size)
        th = [np.random.random() for i in range(X.shape[1])]
        
        iters=1
        
        while iters < n_iter+1:
            for i in range(len(batches)):
                X = batches[i]['x']
                y = batches[i]['y'][0]
                y_pred = np.dot(th,X.T)
                n = len(X)
                for i in range(0,len(th)):
                    D = (-2/n) * sum(X[i] * (y - y_pred))
                    if lr_type=='inverse':
                        th[i] = th[i] - (lr/iters) * D
                    else:
                        th[i] = th[i] - lr * D
            iters = iters+1
            th = th/np.linalg.norm(th)
        self.coef_ = th
        
        return self

    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.th0 = []
        self.th1 = []

        if self.fit_intercept==True:
            X = pd.DataFrame(np.hstack((np.ones((len(X),1)),X)))
        self.X = X
        self.y = y
        X_full = pd.concat([X,y], axis=1, keys=['x','y'])
        X_full = X_full.sample(frac=1, replace=False, random_state=1)
        batches = split(X_full, batch_size)
        th = [np.random.random() for i in range(X.shape[1])]
        
        iters=1
        
        while iters < n_iter+1:
            for i in range(len(batches)):
                X = batches[i]['x']
                y = batches[i]['y'][0]
                y_pred = np.dot(th,X.T)
                # X = X.mul(y-y_pred,axis=0)
                D = np.dot(X.T, y_pred-y)
                # D = (-2/len(X))*X.sum(axis=0,skipna=True)
                if lr_type=='inverse':
                    th = np.subtract(th, (lr/iters)*D)
                else:
                    th = np.subtract(th, lr*D)
            iters = iters+1
            
            th = th/np.linalg.norm(th)
            self.th0.append(th[0])
            self.th1.append(th[1])
        self.coef_ = th
        self.th0 = np.array(self.th0)
        self.th1 = np.array(self.th1)
        
        return self

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        

        if self.fit_intercept==True:
            X = pd.DataFrame(np.hstack((np.ones((len(X),1)),X)))
        self.X = X
        self.y = y
        X_full = pd.concat([X,y], axis=1, keys=['x','y'])
        X_full = X_full.sample(frac=1, replace=False, random_state=1)
        batches = split(X_full, batch_size)
        th = [np.random.random() for i in range(X.shape[1])]
        grad_mse = grad(mse)
        iters=1
        
        while iters < n_iter+1:
            for i in range(len(batches)):
                X = batches[i]['x']
                y = batches[i]['y'][0]
                D = grad_mse(th, X, y)
                # n = len(X)
                for i in range(0,len(th)):
                   
                    if lr_type=='inverse':
                        th[i] = th[i] - (lr/iters) * D[i]
                    else:
                        th[i] = th[i] - lr * D[i]
            iters = iters+1
            th = th/np.linalg.norm(th)
        self.coef_ = th
        
        return self


    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        
        if self.fit_intercept==True:
            X = np.hstack((np.ones((len(X),1)),X))
        thetha = np.dot(pinv(np.dot(X.T,X)),np.dot(X.T, y))
        self.coef_ = thetha
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        if self.fit_intercept == True:
            X = np.hstack((np.ones((len(X),1)),X))
        y_pred = np.dot(self.coef_, X.T)
        return pd.Series(y_pred)

    def give_thetha(self):
        return self.coef_

    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        # self.fit_vectorised(X,y, batch_size=5)

        
        t0 = np.arange(self.th0.min()*0.5, self.th0.max()*2, (self.th0.max() - self.th0.min())/100)
        t1 = np.arange(self.th1.min()*0.5, self.th1.max()*2, (self.th1.max() - self.th1.min())/100)
        th = self.coef_

        t0_mesh, t1_mesh = np.meshgrid(t0, t1)

        error = []
        for i,j in zip(t0_mesh, t1_mesh):
            error.append(np.sum((y.values.reshape(len(y),1) - np.dot(self.X,pd.DataFrame([i,j])))**2, axis=0))
        error = np.array(error)
                
        th[0] = t_0
        th[1] = t_1
        err = mse(th, self.X, self.y)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        
        surf = ax.plot_surface(t0_mesh, t1_mesh, error, cmap='viridis', edgecolor='none', alpha=0.6)
        ax.scatter(t_0, t_1, err, color='r')
        # ax.set_title('Surface plot')
        ax.view_init(azim=-55, elev=50)
        ax.set_zlim([0,800])
        # plt.zlim([0,800])
        plt.xlim([0, 2])
        plt.ylim([0,2])
        plt.xlabel("Thetha 0")
        plt.ylabel("Thetha 1")
        ax.set_zlabel("Error")
        plt.title("Error = "+str(err))
        plt.colorbar(surf, shrink=0.5, aspect=5)
        print("plotted")
        return fig

    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        self.fit_vectorised(X,y, batch_size=5)
        th = self.coef_
        fig = plt.figure()
        ax = plt.axes()
        plt.scatter(X, y, color='b')
        plt.title("th1 = "+str(th[1])+"      th0 = "+str(th[0]))
        th[0] = t_0
        th[1] = t_1
        y_pred = np.dot(th, self.X.T)
        plt.plot(X, y_pred, color='r')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.xlim([-2,2])
        plt.ylim([-2,2])
        # plt.show()
        print("plotted")
        return fig

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
                
        t0 = np.arange(self.th0.min()*0.5, self.th0.max()*2, (self.th0.max() - self.th0.min())/100)
        t1 = np.arange(self.th1.min()*0.5, self.th1.max()*2, (self.th1.max() - self.th1.min())/100)
        th = self.coef_

        t0_mesh, t1_mesh = np.meshgrid(t0, t1)

        error = []
        for i,j in zip(t0_mesh, t1_mesh):
            error.append(np.sum((y.values.reshape(len(y),1) - np.dot(self.X,pd.DataFrame([i,j])))**2, axis=0))
        error = np.array(error)
                
        th[0] = t_0
        th[1] = t_1
        err = mse(th, self.X, self.y)

        fig = plt.figure()
        ax = plt.axes()

        ax.contour(t0_mesh, t1_mesh, error, 20)
        
        ax.scatter(t_0, t_1, color='r')
        # ax.set_title('Contour plot')
        # ax.set_zlim([0,800])
        plt.xlim([0, 2])
        plt.ylim([0, 2])
        plt.xlabel("thetha 0")
        plt.ylabel("thetha 1")
        plt.title("Error = "+str(err))
        print("plotted")
        return fig



        
