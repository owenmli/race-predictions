import numpy as np
import csv



X = []
with open('parsedData.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append(row)

Y = []
with open('attendance.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Y.append(row)

X = X[1:28]
Y = Y[1:28]

X = np.array(X)
Y = np.array(Y)
W = np.zeros(4)

print W


def logistic_func(weights, x):
    return float(1) / (1 + math.e**(-x.dot(weights)))



