import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets
import csv

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

lr = LogisticRegression(C=10000.0, random_state=0)
lr.fit(X_train_std, y_train)

output = []

print(lr.predict_proba(X_test_std[:,:]))
for i in lr.predict_proba(X_test_std[:,:]):
	output.append([i.argmax()])

with open("output10000.csv", "w", newline='') as f:
	writer = csv.writer(f)
	writer.writerows(output)


