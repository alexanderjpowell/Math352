import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import statsmodels.api as sm

train_filename = 'Problem8_train.csv'
test_filename = 'Problem8_test.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train.values[:,0:37]
y_train = train.values[:,37]
X_test = test.values[:,1:38]

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

result = sm.OLS(y_train, X_train).fit()

print(result.summary())

y_pred = result.predict(X_test)
import numpy as np
np.savetxt("revenue_pred.csv", y_pred, delimiter=",")