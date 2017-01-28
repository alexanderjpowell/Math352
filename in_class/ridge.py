import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df[['CRIM', 'ZN', 'LSTAT', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B']].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
'''
# Ridge
RR = linear_model.RidgeCV(alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train, y_train)
best_lambda = RR.alpha_
print("Lambda #1: " + str(best_lambda))
#

# Lasso
RR = linear_model.LassoCV(alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train, y_train)
best_lambda = RR.alpha_
print("Lambda #1: " + str(best_lambda))
#
'''
# Elastic.Net
RR = linear_model.ElasticNetCV(l1_ratio=0.5, alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train, y_train)
best_lambda = RR.alpha_
print("Lambda #1: " + str(best_lambda))
#

y_test_pred = RR.predict(X_test)
mse1 = mean_squared_error(y_test, y_test_pred)
print("MSE #1: " + str(mse1))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
'''
# Ridge
RR = linear_model.RidgeCV(alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train_std, y_train)
lambda2 = RR.alpha_
print("Lambda #2: " + str(lambda2))
#

# Lasso
RR = linear_model.LassoCV(alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train_std, y_train)
lambda2 = RR.alpha_
print("Lambda #2: " + str(lambda2))
#
'''
# Elastic.Net
RR = linear_model.ElasticNetCV(l1_ratio=0.5, alphas = [0.01, 0.05, 0.06, 0.1, 0.5, 1, 5, 10])
RR.fit(X_train_std, y_train)
lambda2 = RR.alpha_
print("Lambda #2: " + str(lambda2))
#

y_test_pred = RR.predict(X_test_std)
mse2 = mean_squared_error(y_test, y_test_pred)
print("MSE #2: " + str(mse2))





