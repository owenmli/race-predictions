import numpy as np
import csv
import math
from decimal import *
import matplotlib.pyplot as plt
import random



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

X = X[0:]
Y = Y[0:]

X = np.matrix(X)
Y = np.matrix(Y)
W = np.zeros(5)
W = np.matrix(W)
W = np.transpose(W)





X = X.astype(np.float)
Y = Y.astype(np.float)
W = W.astype(np.float)


def logistic_func(WT, x):
	x = np.transpose(x)
	WTx = np.dot(WT,x)
	WTx = np.asscalar(WTx)
	try:
		power = math.pow(math.e,-WTx)
	except Exception, e:
		return 2.2250738585072014e-308
	sigmoid = 1.0 / (1.0 + power)
	if sigmoid == 1.0:
		return 0.9999999999
	else:
		return sigmoid

def error_func(W, X, Y):
	totalError = 0.0
	WT = np.transpose(W)
	for(x,y) in zip(X,Y):
		sigmoid = logistic_func(WT,x)
		totalError += (np.asscalar(y) * math.log(sigmoid)) + ((1.0-np.asscalar(y)) * math.log(1.0 - sigmoid))
	return -totalError

def gradient_desc(W, X, Y, alpha=.00015, converge_change= 0.5):
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	error = error_func(W, X, Y)
	change_cost = 1
	i=0
	while(change_cost > converge_change):
		old_error = error
		W = W + (alpha * (gradient(W, X, Y)))
		error = error_func(W, X, Y)
		change_cost = old_error - error
		print "Iteration: " + str(i) + " Delta: " + str(change_cost) + " Error: " + str(error)
		i+=1

	return W

def gradient(W, X, Y):

	# sumOfVectors = np.zeros(4)
	# sumOfVectors = np.matrix(W)
	sumOfVectors = 0.0

	WT = np.transpose(W)

	for(x,y) in zip(X,Y):
		subtraction = np.asscalar(y) - logistic_func(WT, x)
		xt = np.transpose(x)
		partialGradProduct = np.dot(xt,subtraction)
		sumOfVectors += partialGradProduct

	return sumOfVectors

def calcAccuracy(W,X,Y):
	counter = 0
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	WT = np.transpose(W)
	for (xi,y) in zip(X,Y):
		result = logistic_func(WT,xi)
		print result * 100
		if (result > 0.5 and y == 1) or (result <= 0.5 and y == 0):
			counter += 1


	successrate = 100 * (float(counter)/float(len(X)))
	print "Accuracy Rate: " + str(successrate) + "%"


# Wtest = np.zeros(5)
# Wtest = np.matrix(Wtest)
# Wtest = np.transpose(Wtest)


def generateNFoldWeights(n, X, Y, W):

	combined = list(zip(X,Y))
	random.shuffle(combined)
	X[:], Y[:] = zip(*combined)

	nlength = float(len(X))/n
	nX = X[0:int(nlength)]
	nY = Y[0:int(nlength)]
	tX = X[int(nlength)+1:]
	tY = Y[int(nlength)+1:]

	W = gradient_desc(W,nX,nY)
	calcAccuracy(W,tX,tY)
	#calcAccuracy(W, nX, nY)

	# W = gradient_desc(W,nX, nY)
	# calcAccuracy(W, nX,nY)
	# calcAccuracy(W, tX, tY)



generateNFoldWeights(2,X,Y,W)

