import pandas as pd
import statsmodels.api as sm


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())


#X = df[['RM', 'CRIM']].values
X = df.iloc[:,:-1].values


y = df[['MEDV']].values
#X = sm.add_constant(X)
result = sm.OLS(y, X).fit()
print(result.summary())

print("R2:  ", result.rsquared)
print("MSE: ",result.mse_model)
#print("Predicted Values: ", result.predict())
X_test = X[1:7,:]
#print("Predicted Values: ", result.predict(X_test))