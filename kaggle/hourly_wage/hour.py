import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

train_filename = 'Income_training.csv'
test_filename = 'Income_testing.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train[['age', 'yearsEducation', 'sex1M0F']].values
y_train = train[['compositeHourlyWages']].values
X_test = test[['age', 'yearsEducation', 'sex1M0F']].values

X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

result = sm.OLS(y_train, X_train).fit()

print(result.summary())

y_pred = result.predict(X_test)
import numpy as np
np.savetxt("OLS_pred.csv", y_pred, delimiter=",")