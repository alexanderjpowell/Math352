from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

X = [[0],[1],[2],[3]]
Y = [0,0,1,1]

knnRegressor = KNeighborsRegressor(n_neighbors = 2, weights = 'uniform')
knnRegressor.fit(X, Y)
print(knnRegressor.predict([[1.5]]))

# solution is 0.5 because halfway between the 1 and 2 in X
# corresponds to halfway between 0 and 1 in Y
# (0+1)/2 = 1/2