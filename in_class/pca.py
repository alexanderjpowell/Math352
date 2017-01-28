import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
X, y = df_wine.iloc[:,1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

pca = PCA()
X_train_pca_sklearn = pca.fit_transform(X_train_std)
variance_retained = 0.9
cum_VE = 0
i = 1
while (cum_VE < variance_retained):
	i = i + 1
	cum_VE = sum(pca.explained_variance_ratio_[0:i])
	npcs = i

print("Use " + str(npcs) + " principle components to retain " + str(variance_retained * 100) + " percent of the variance.")
