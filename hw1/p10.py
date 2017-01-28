# Homework 1, Problem 10
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == "__main__":

	# Part A and B - Create the scatterplot
	iris = datasets.load_iris()

	# Generate the scatterplot
	X1 = iris.data[:,3] # Petal width
	X2 = iris.data[:,2] # Petal length
	Y = -X1 + 4

	plt.scatter(X1[iris.target == 2], X2[iris.target == 2], color = 'red')
	plt.scatter(X1[iris.target == 0], X2[iris.target == 0], color = 'blue')
	plt.plot(X1, Y, color = 'black', linewidth = 3)

	plt.show()

	# Part C - Calculate the CE
	Y_hat = X2 < -X1 + 2
	Y = iris.target == 0

	CE = (sum(Y[0:50] != Y_hat[0:50]) + sum(Y[100:] != Y_hat[100:])) / 100

	print("The CE is: " + str(CE))