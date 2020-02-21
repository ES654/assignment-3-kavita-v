import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

N_list = [20, 30, 40, 50]
d_list = [1, 3, 5, 7, 9]

for i in range(len(N_list)):
    plt.subplot(1,len(N_list), i+1)
    n = N_list[i]
    plt.xlabel('Degree')
    plt.ylabel('Max Absolute thetha_i')
    plt.title('N='+str(n))
#     plt.ylim([0,250])
    plt.xlim([0,10])
    plt.ylim([0.4,1])

    x = np.array([i*np.pi/180 for i in range(1,n*5,5)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    y = pd.Series(y)
    x = x.reshape(n,1)

    LR = LinearRegression(fit_intercept=False)
    X_plot = []
    y_plot = []
    y2_plot = []
    for d in d_list:
        poly = PolynomialFeatures(d, include_bias=True)
        x_new = pd.DataFrame(poly.transform(x))
        # print(x_new)
        LR.fit_vectorised(x_new, y, batch_size=7, lr_type='inverse')
        th = LR.give_thetha()
        # print(th)
        th_max = max(abs(th))
        th_magn = np.linalg.norm(th)
        y_plot.append(th_max)
        y2_plot.append(th_magn)
        # print(th_magn)
        X_plot.append(d)

        plt.plot(X_plot, y_plot)
# plt.xlim([0,7])
# plt.xlabel('Degree')
# plt.ylabel('Max Absolute Value of thetha_i')
# plt.show()

        # plt.plot(X_plot, np.log(y2_plot))
        # plt.plot(X_plot, y2_plot)
        # plt.xlim([0,7])
        
        
plt.show()
