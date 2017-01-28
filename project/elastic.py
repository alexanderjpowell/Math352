'''
Elastic.NET Regression for the forest fire data set
'''

import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

forest_data = pd.read_csv('forestfires.csv', header = 0)

X = forest_data.iloc[:,4:8].values
y = forest_data.iloc[:,12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

RR = linear_model.ElasticNetCV(l1_ratio=0.5, alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train, y_train)
best_lambda = RR.alpha_
print("Lambda: " + str(best_lambda))

y_test_pred = RR.predict(X_test_std)
mse2 = mean_squared_error(y_test, y_test_pred)
print("MSE: " + str(mse2))