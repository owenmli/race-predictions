import numpy as np
import csv
import math
from decimal import *



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

X = X[1:3]
Y = Y[1:3]

X = np.matrix(X)
Y = np.matrix(Y)
W = np.zeros(4)
W = np.matrix(W)
W = np.transpose(W)


X = X.astype(np.float)
Y = Y.astype(np.float)
W = W.astype(np.float)


def logistic_func(W, x):
	wt = np.transpose(W)
	x = np.transpose(x)
	dot = np.dot(wt,x)
	#dot = np.squeeze(dot)
	dot = np.asscalar(dot)
	result = 1.0 / (1.0 + math.pow(math.e,-dot))
	return result

def error_func(W, X, Y):
	errorSumP1 = 0.0
	errorSumP2 = 0.0
	totalError = 0.0
	for(x,y) in zip(X,Y):
		sigmoid = logistic_func(W,x)
    	errorSumP1 = np.asscalar(y) * math.log(sigmoid)
    	errorSumP2 = (1.0-np.asscalar(y)) * math.log(1.0 - sigmoid)
    	totalError = totalError + errorSumP1 + errorSumP2
	return -totalError

def gradient_desc(W, X, Y, alpha=.001, converge_change=.001):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    error = error_func(W, X, Y)
    change_cost = 1
    i=0
    while(change_cost > converge_change):
    	print "Iteration: " + str(i)
        old_error = error
        W = W + (alpha * gradient(W, X, Y))
        print "Weights: " + str(W)
        error = error_func(W, X, Y)
        change_cost = old_error - error
        i+=1
    return W

def gradient(W, X, Y, alpha=.001):

	partialSum = np.zeros(4)
	partialSum = np.matrix(W)

	for(x,y) in zip(X,Y):
		print x
		print W
		print "SIGMOID: " + str(logistic_func(W,X))
		subtraction = np.asscalar(y) - logistic_func(W, x)
		print "SUBTRACTION: " + str(subtraction)
		print (np.dot(np.transpose(x),subtraction))
		print "PARTIAL SUM1 : " + str(partialSum)
		partialSum = partialSum + np.dot(np.transpose(x),subtraction)
		print "PARTIAL SUM: " + str(partialSum)

	return partialSum


W = gradient_desc(W,X,Y)
print W