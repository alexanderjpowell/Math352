import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

train_filename = 'train.csv'
test_filename = 'test.csv'

train = pd.read_csv(train_filename)
test = pd.read_csv(test_filename)

X_train = train.values[:,1:785]
y_train = train.values[:,0]
X_test = test.values[:,0:785]

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

pca = PCA()
X_train_pca_sklearn = pca.fit_transform(X_train_std)
variance_retained = 0.99
cum_VE = 0
i = 1
while (cum_VE < variance_retained):
	i = i + 1
	cum_VE = sum(pca.explained_variance_ratio_[0:i])
	npcs = i

print("Use " + str(npcs) + " principle components to retain " + str(variance_retained * 100) + " percent of the variance.")
#pca = PCA(n_components = npcs)
#X_train_pca = pca.fit_transform(X_train_std)
#X_test_pca = pca.transform(X_test_std)





cov_mat = np.cov(X_train_std.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Reverse sort the eigenvalues
eigen_vals = sorted(eigen_vals, reverse=True)

print('------------------------------------------------')
total = sum(eigen_vals)
print('Sum: ' + str(total))
running_total = 0
count = 1
for i in eigen_vals:
	running_total = running_total + i
	if (running_total / total >= 0.99):
		print(str(count) + ' to reach 99 percent')
		break
	count = count + 1

total = sum(eigen_vals)
var_exp = eigen_vals/total
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 785), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1, 785), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principle components')
plt.legend(loc='best')
plt.show()

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

eigen_pairs = sorted(eigen_pairs, reverse=True)
#eigen_pairs.sort(reverse=True)

w = np.hstack((eigen_pairs[0][1][:,np.newaxis], eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n', w)

X_train_std[0].dot(w)

X_train_pca = X_test_std.dot(w)

colors = ['r', 'g', 'b']

markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
	plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

pca = PCA(n_components = 3)
knn = KNeighborsClassifier(n_neighbors = 3, p=2, metric='minkowski')
X_train_pca_sklearn = pca.fit_transform(X_train_std)
X_test_pca_sklearn = pca.transform(X_test_std)
knn.fit(X_train_pca_sklearn, y_train)
y_pred_sklearn = knn.predict(X_test_pca_sklearn)
















