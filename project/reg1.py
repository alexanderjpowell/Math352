import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

train_filename = 'fire_training.csv'
test_filename = 'fire_testing.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train[['temp', 'RH', 'wind', 'rain']].values
y_train = train[['area']].values

X_test = test[['temp', 'RH', 'wind', 'rain']].values