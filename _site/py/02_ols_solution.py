from sklearn.datasets import make_regression
# http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html#sklearn.datasets.make_regression
from time import time
import numpy as np
# simple exemple 10 echantillons , 2 predicteurs
X, y = make_regression(n_samples=10, n_features=2)

# OLS, moindre carrés
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# prediction
yhat = X[:, 0]* beta[0] + X[:,1] * beta[1]

# erreur residuelle
e = y - yhat

print("l'erreur residuelle moyenne est {:.2f} ".format(mean(y-yhat)))

# maintenant le meme exemple avcec un autre ordre de grandeur
N = 100000
M = 1000
X, y = make_regression(n_samples=N, n_features=M)
t = time()
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("Calcul fait en {:.2f}s".format( (time() -t) ))

# prediction
yhat = [0 for i in range(N)]
for k in range(M):
    yhat += X[:, k]* beta[k]

# erreur residuelle
e = y - yhat

print("l'erreur residuelle moyenne est {:.2f} ".format(mean(y-yhat)))

# maintenant avec du bruit gaussien de variance 10
N = 1000
M = 10
X, y = make_regression(n_samples=N, n_features=M, noise =10)
t = time()
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print("Calcul fait en {:.2f}s".format( (time() -t) ))

# prediction
yhat = [0 for i in range(N)]
for k in range(M):
    yhat += X[:, k]* beta[k]

# erreur residuelle
e = np.abs(y - yhat)

print("l'erreur residuelle moyenne est {:.2f} ".format(mean(e)))
plt.plot(e)
