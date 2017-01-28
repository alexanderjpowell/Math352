# Homework 1, Problem 9
import matplotlib.pyplot as plt
from sklearn import datasets

if __name__ == "__main__":
	# Part A and B - Create the scatterplot
	iris = datasets.load_iris()

	X = iris.data[:,1] # sepal width
	Y = iris.data[:,3] # petal width
	Y_hat = -0.75 * X + 3.5

	plt.scatter(X, Y) # Generate the scatterplot
	plt.plot(X, Y_hat, color = 'black', linewidth = 3) # Plot the line
	plt.show() # Display the scatterplot

	# Part C - Calculate the MSE
	diff = []
	count = 0
	for i in Y: # Subtract to get error
		diff.append(abs(Y[count] - Y_hat[count]) ** 2)
		count += 1
	MSE = sum(diff) / len(diff) # Finally, calculte the MSE
	print("The mean squared error is: " + str(MSE))