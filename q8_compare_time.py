import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
import time


plt.subplot(2,1,1)
X_plot=[]
y_gd = []
y_norm = []
for n in range(20,200,20):
    x = np.array([i*np.pi/180 for i in range(1,n*5*3,5)])
    np.random.seed(10)  #Setting seed for reproducibility
    
    x = x.reshape(n,3)
    y = 4*x.T[0] + 7 + np.random.normal(0,3,len(x))
    y = pd.Series(y)
    x = pd.DataFrame(x)
    LR = LinearRegression(fit_intercept=False)
    LR2 = LinearRegression(fit_intercept=False)
    st1 = time.time()
    LR2.fit_normal(x,y)
    
    st2 = time.time()
    LR.fit_autograd(x, y, batch_size=7, lr_type='inverse')
    st3 = time.time()
    y_gd.append(st3 - st2)
    y_norm.append(st2 - st1)
    X_plot.append(n)

plt.plot(X_plot,y_gd, 'r--', label='Gradient Descent')
plt.plot(X_plot,y_norm, 'b--', label='Normal Method')
plt.xlabel('N')
plt.ylabel('Time')
plt.title('Comparing time required with change in N')
plt.legend()
# plt.ylim([0,250])
# plt.xlim([0,10])


plt.subplot(2,1,2)
X_plot=[]
y_gd = []
y_norm = []
for d in range(1, 10, 1):
    x = np.array([i*np.pi/180 for i in range(1,500*d*5,5)])
    np.random.seed(10)  #Setting seed for reproducibility
    
    x = x.reshape(500,d)
    y = 4*x.T[0] + 7 + np.random.normal(0,3,len(x))
    y = pd.Series(y)
    x = pd.DataFrame(x)
    LR = LinearRegression(fit_intercept=False)
    LR2 = LinearRegression(fit_intercept=False)
    st1 = time.time()
    LR.fit_autograd(x, y, batch_size=7, lr_type='inverse')
    st2 = time.time()
    LR2.fit_normal(x,y)
    st3 = time.time()
    y_gd.append(st2 - st1)
    y_norm.append(st3 - st2)
    X_plot.append(d)

plt.plot(X_plot,y_gd, 'r--', label='Gradient Descent')
plt.plot(X_plot,y_norm, 'b--', label='Normal Method')
plt.xlabel('degree')
plt.ylabel('Time')
plt.title('Comparing time required with change in degree')      
        
plt.show()
