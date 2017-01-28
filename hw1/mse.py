import math

if __name__ == "__main__":

	y = [3, 1, 3.5, 1]
	y_hat = [2.5, 1.5, 3, 0]

	y_diff = []
	for i in range(0,4):
		y_diff.append(abs(y[i] - y_hat[i]))

	print("y_diff:             " + str(y_diff))
	sum_of_squares = 0
	for i in y_diff:
		sum_of_squares += (i ** 2)

	print("sum_of_squares:     " + str(sum_of_squares))
	print("Mean Squared Error: " + str(sum_of_squares / 4))