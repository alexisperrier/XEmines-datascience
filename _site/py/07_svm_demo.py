import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
import matplotlib.pyplot as plt

# -----------------------------------------------------
#  Linearly separable
# -----------------------------------------------------
def plot_hyperplane(clf, min_x, max_x, label):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - distance, max_x +distance)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, label = label)

def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)

# Build dataset, 200 points, 2 dimensions, 2 classes
distance = 1
X = np.random.randn(200,2)
X[:150] = X[:150] -distance
X[151:] = X[151:] +distance

# first 150 point belong to class -1
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3,
    shuffle = True,
    random_state=2)

plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)
plt.xlabel('X1')
plt.ylabel('X2');

parameters = {'C': [0.01, 0.1, 1, 10, 100]}

for C in parameters['C']:
    clf = SVC(kernel='linear', C = C)
    clf.fit(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    acc_train = clf.score(X_train, y_train)
    print(" C: {} train {:.4f} test  {:.4f}".format(C, acc_train, acc_test) )
    plot_hyperplane(clf, min_x, max_x, "C: {}".format(C))

plt.legend()
plt.show()

# -----------------------------------------------------
#  Non Linearly separable
# -----------------------------------------------------

distance = 1
X = np.random.randn(200,2)
X[:100] = X[:100] -distance
X[101:150] = X[101:150] +distance

# first 150 point belong to class -1
y = np.concatenate([np.repeat(-1, 150), np.repeat(1,50)])

min_x = np.min(X[:, 0])
max_x = np.max(X[:, 0])

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.3,
    shuffle = True,
    random_state=2)

fig,ax = plt.subplots(1,1, figsize=(8,8))
plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)

plt.xlabel('X1')
plt.ylabel('X2');

parameters = {'C': [0.01, 0.1, 1, 10, 100]}

# kernel = poly, rbf, sigmoid
for C in parameters['C']:
    clf = SVC(kernel='sigmoid', C = C)
    clf.fit(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    acc_train = clf.score(X_train, y_train)
    print(" C: {} train {:.4f} test  {:.4f}".format(C, acc_train, acc_test) )


parameters = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.5, 1,2,3,4]}
clf = GridSearchCV(SVC(kernel='rbf'), parameters, cv=10, scoring='accuracy')
clf.fit(X, y)

clf.best_score_
clf.best_params_
clf = clf.best_estimator_

fig,ax = plt.subplots(1,1, figsize=(8,8))
plt.scatter(X[:,0], X[:,1], s=70, c=y, cmap=plt.cm.Paired)

plt.xlabel('X1')
plt.ylabel('X2');

plot_svc(clf, X, y)
# plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=10)

plt.legend()
plt.show()

# grid search
