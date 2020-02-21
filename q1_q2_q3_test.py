
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
# from linearRegression.linearRegression import *
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

print("Non-vectorised")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y, batch_size=7, lr_type='inverse') # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('For fit_intercept = '+str(fit_intercept))
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

print("Autograd")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('For fit_intercept = '+str(fit_intercept))
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))

print("Vectorised")
for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y, batch_size=7) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('For fit_intercept = '+str(fit_intercept))
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
