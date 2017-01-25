import numpy as np
import csv
import math
from decimal import *
import matplotlib.pyplot as plt
import random


def getCSVAndFormat(CSVF_name):
	targetMatrix = []
	temp = []
	with open(CSVF_name,'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			temp.append(row)

	temp = temp[1:]

	for entry in temp:
		entry = [float(i) for i in entry]
		entry.pop(0)
		targetMatrix.append(entry)

	targetMatrix = np.matrix(targetMatrix)
	targetMatrix = targetMatrix.astype(np.float)

	return targetMatrix


#Training 
X = getCSVAndFormat('trainX.csv')
Y = getCSVAndFormat('trainY.csv')

#Validation
Xval = getCSVAndFormat('valX.csv')
Yval = getCSVAndFormat('valY.csv')


W = np.zeros(6)
W = np.matrix(W)
W = np.transpose(W)
W = W.astype(np.float)

Z = []

training_error = []
validation_error = []



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

def gradient_desc(W, X, Y, alpha=.000015, converge_change= 0.25):
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	error = error_func(W, X, Y)
	change_cost = 1
	i=0
	while(change_cost > converge_change):
		old_error = error
		W = W + (alpha * (gradient(W, X, Y)))
		error = error_func(W, X, Y)
		training_error.append(error)
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
	correctComing = 0
	guessComing = 0
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	WT = np.transpose(W)
	error = error_func(W,X,Y)
	for (xi,y) in zip(X,Y):
		result = logistic_func(WT,xi)
		if(result >= 0.5):
			Z.append(1)
			guessComing += 1
		else:
			Z.append(0)

		if result >= 0.5 and y==1:
			correctComing += 1

		if (result >= 0.5 and y == 1) or (result < 0.5 and y == 0):
			counter += 1

	successrate = 100 * (float(counter)/float(len(X)))
	print "Guessed " + str(guessComing) + " are coming, " + str(correctComing) + " are correct."
	print "Accuracy Rate: " + str(successrate) + "%"
	print_predicted_attendance(Z,Y)


def main(X, Y, W):

	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]

	W = gradient_desc(W,X,Y)
	calcAccuracy(W,X,Y)
	print W



def print_predicted_attendance(Z,Y):
	with open("results.csv","w") as f:
	    wr = csv.writer(f,delimiter="\n")
	    wr.writerow(Z)

	with open("testhalf.csv","w") as f:
		wr = csv.writer(f,delimiter="\n")
		wr.writerow(Y)



main(X,Y,W)

