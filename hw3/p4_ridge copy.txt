import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np

train_filename = 'Problem8_train.csv'
test_filename = 'Problem8_test.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train.values[:,0:37]
y_train = train.values[:,37]
X_test = test.values[:,1:38]

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

RR = linear_model.RidgeCV(alphas = [0.01, 0.1, 1, 5, 10, 25, 50])
RR.fit(X_train_std, y_train)
lambda2 = RR.alpha_
print("Lambda : " + str(lambda2))


y_test_pred = RR.predict(X_test_std)
np.savetxt("revenue_pred_ridge.csv", y_test_pred, delimiter=",")





