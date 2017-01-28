import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

train_filename = 'Problem7_train.csv'
test_filename = 'Problem7_test.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train.values[:,1:785]
y_train = train.values[:,0]
X_test = test.values[:,1:785]

print(len(X_train))
print(len(y_train))
print(len(X_test))

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(y_pred)


np.savetxt("mnist_pred.csv", y_pred, delimiter=",")