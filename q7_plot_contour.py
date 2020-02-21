import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))



LR = LinearRegression(fit_intercept=True)
LR = LR.fit_vectorised(X, y, batch_size=5)
th0 = LR.th0
th1 = LR.th1

for i in range(5,55,5):
    fig1 = LR.plot_surface(X, y, th0[i],th1[i])
    fig1.savefig('surfplot/iteration{}.png'.format(i))
    fig2 = LR.plot_line_fit(X, y, th0[i], th1[i])
    fig2.savefig('lineplot/iteration{}.png'.format(i))
    fig3 = LR.plot_contour(X, y, th0[i], th1[i])
    fig3.savefig('contourplot/iteration{}.png'.format(i))
