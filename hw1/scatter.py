import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	N = 50
	x = [1, 2, 4, 0]
	y = [0.5, 1, 2, 0]
	plt.scatter(x, y)
	plt.show()



'''
plt.scatter(X[iris.target==0], Y[iris.target==0], color='red')
plt.scatter(X1[iris.target==1], X2[iris.target==1], color='blue')
plt.plot(X1, -X1+3, color='black', linewidth=3)
plt.show()
'''

'''
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
#print(iris.data[:,0])
#print(iris.data[:,1])
#rint(iris.data[:,2])
#rint(iris.data[:,3])

X = iris.data[:,2]
Y = iris.data[:,3]

plt.scatter(X, Y, color='black')
plt.plot(X, 0.5*X-0.5, color='blue', linewidth=3)
plt.show()
'''

'''
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# Generate the scatterplot
	#x = [1, 2, 4, 0]
	#y = [0.5, 1, 2, 0]
	#plt.scatter(x, y)
	#plt.show()

	# Plot the line
	x = np.arange(10) # Change the dimensions accordingly
	y = -0.75 * x + 3.5
	m, b = np.polyfit(x, y, 1)
	plt.plot(x, m*x + b, '-')
	plt.show()

	# Calculate the MSE
	'''