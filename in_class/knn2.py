from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

iris = datasets.load_iris()
X = iris.data[:,2]
Y = iris.data[:,3]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

knnRegressor = KNeighborsRegressor(n_neighbors = 1, weights = 'uniform')

knnRegressor.fit(X_train.reshape(105,1), Y_train)
Y_pred = knnRegressor.predict(X_train.reshape(105, 1))
print(mean_squared_error(Y_pred, Y_train))