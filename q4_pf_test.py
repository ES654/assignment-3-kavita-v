import numpy as np
from preprocessing.polynomial_features import PolynomialFeatures



X = np.array([[1, 2, 5],[3, 4, 6]])
poly = PolynomialFeatures(3, include_bias=False)
poly = poly.transform(X)
print(poly)
