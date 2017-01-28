"""
OLS for Data Analysis Final Project
"""

# Set up Python libraries
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import statsmodels.api as sm


# Import Data
forest_data = pd.read_csv('forestfires.csv', header = 0)
y = forest_data.iloc[:,12].values
X = forest_data.iloc[:,8:12].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Add constants for regression
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Normalize the data
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# Run OLS
result = sm.OLS(y_train, X_train_std).fit()
y_pred = result.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)

# Correlation 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid',context='notebook')
cols = ['FFMC','DMC','DC','ISI']
sns.pairplot(forest_data[cols],size=2.5)
plt.show()

# correlation matrix 
cm = np.corrcoef(forest_data[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15}, yticklabels = cols,xticklabels=cols)
plt.show()
