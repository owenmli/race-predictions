import numpy as np
import csv
import math



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

X = np.matrix(X)
Y = np.matrix(Y)
W = np.zeros((4,1))


X = X.astype(np.float32)
Y = Y.astype(np.float32)
W = W.astype(np.float32)

def logistic_func(W, x):
	wt = np.transpose(W)
	x = np.transpose(x)
	dot = np.dot(wt,x)
	print (1) / (1 + math.pow(math.e,-dot))

def error_func(W, X, Y):
	errorSumP1 = 0
	errorSumP2 = 0
	totalError = 0
	for(x,y) in zip(X,Y):
		sigmoid = logistic_func(W,x)
    	errorSumP1 = y * math.log(sigmoid)
    	errorSumP2 = (1-y) * math.log(1 - sigmoid)
    	totalError = totalError + errorSumP1 + errorSumP2
    return totalError

def gradient_desc(weights, X, Y, alpha=.001, converge_change=.001):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    error = error_func(weights, X, Y)
    change_cost = 1
    while(change_cost > converge_change):
        old_error = error
        weights = weights - (alpha * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        change_cost = old_cost - cost
    return weights

def gradient(weights, X, Y):
	for(x,y) in zip(X,Y):
		sigmoid = logistic_func(weights, x) - np.squeeze(y)
		final_calc = first_calc.T.dot(x)
		return final_calc
