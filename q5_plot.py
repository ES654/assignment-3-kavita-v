import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))
y = pd.Series(y)
x = x.reshape(60,1)

LR = LinearRegression(fit_intercept=False)
X_plot = []
y_plot = []
y2_plot = []
for d in range(1, 15):
    poly = PolynomialFeatures(d, include_bias=True)
    x_new = pd.DataFrame(poly.transform(x))
    # print(x_new)
    LR.fit_vectorised(x_new, y, batch_size=7, lr_type='inverse')
    th = LR.give_thetha()
    # print(d)
    # print(th)
    th_max = max(abs(th))
    th_magn = np.linalg.norm(th)
    y_plot.append(th_max)
    y2_plot.append(th_magn)
    # print(th_magn)
    X_plot.append(d)

plt.plot(X_plot, y_plot)
# plt.xlim([0,7])
plt.xlabel('Degree')
plt.ylabel('Max Absolute Value of thetha_i')
plt.ylim([0.7,1])
plt.show()

# plt.plot(X_plot, y2_plot )
# # plt.xlim([0,7])
# plt.xlabel('Degree')
# plt.ylabel('Magnitude of thetha on log scale')
# plt.show()
