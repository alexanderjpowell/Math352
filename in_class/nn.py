import time
from sknn.mlp import Classifier, Layer
import pandas as pd

data_test = pd.read_csv('test.csv')
data_train = pd.read_csv('train.csv')

y_train = data_train.iloc[:,0]
X_train = data_train.iloc[:,1:]/255
X_test = data_test.iloc[:,:]/255

nn = Classifier(Layers=[Layer("Sigmoid", units=20), Layer("Softmax", units=10)], learning_rate=0.001, n_iter=25)

now = time.time()
nn.fit(X_train[0:1000], y_train[0:1000])
print(time.time() - now)
y_pred2 = nn.predict(X_train[0:2001])
performance = sum([[y_train[1001:2001]]]==y_pred2)/1000
print(performance)
#np.savetxt('pred.csv', y_pred2, delimiter=',')