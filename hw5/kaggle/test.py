import csv

a = [[2],[4],[1],[9],[5]]
#a = [[2,3],[4,5],[6,7]]

with open("output.csv", "w", newline='') as f:
	writer = csv.writer(f)
	writer.writerows(a)

	