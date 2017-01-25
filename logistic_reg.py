import numpy as np
import csv
import math
from decimal import *
import matplotlib.pyplot as plt
import random

#Function to parse CSV and transform them into numpy matricies
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

#Weights
W = np.zeros(6)
W = np.matrix(W)
W = np.transpose(W)
W = W.astype(np.float)

Z = []

training_error = []


#Calculating sigmoid
def logistic_func(WT, x):
	x = np.transpose(x)
	WTx = np.dot(WT,x)
	WTx = np.asscalar(WTx)
	try:
		power = math.pow(math.e,-WTx)
	except Exception, e:
		#To avoid math domain errors, we never want sigmoid to reach 1 or 0, we instead return a number very close to 1 or very close to 0
		return 2.2250738585072014e-308
	sigmoid = 1.0 / (1.0 + power)
	if sigmoid == 1.0:
		return 0.9999999999
	else:
		return sigmoid

#Calculating the error function in logistic regression
def error_func(W, X, Y):
	totalError = 0.0
	WT = np.transpose(W)
	#iterate through each "person/row" to sum the total error
	for(x,y) in zip(X,Y):
		sigmoid = logistic_func(WT,x)
		totalError += (np.asscalar(y) * math.log(sigmoid)) + ((1.0-np.asscalar(y)) * math.log(1.0 - sigmoid))
	return -totalError

#applying gradient descent on the derived error function to train weights
def gradient_desc(W, X, Y, alpha=.00015, converge_change= 0.5):
	#normalize data
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	#Add column of ones to data to include intercept and get weight w0
	X = np.c_[np.ones(X.shape[0]),X]
	error = error_func(W, X, Y)
	change_cost = 1
	i=0
	#We continue iteration until our error is below a tolerance threshold. We define delta as the OldError - NewError. When delta is sufficiently small, stop iteration.
	while(change_cost > converge_change):
		old_error = error
		W = W + (alpha * (gradient(W, X, Y)))
		error = error_func(W, X, Y)
		training_error.append([i,error])
		change_cost = old_error - error
		print "Iteration: " + str(i) + " Delta: " + str(change_cost) + " Error: " + str(error)
		i+=1
	return W

#Calculate the term we will be multiplying our learning rate alpha by.
def gradient(W, X, Y):
	sumOfVectors = 0.0
	WT = np.transpose(W)
	#Need to iterate through each "person/row"
	for(x,y) in zip(X,Y):
		subtraction = np.asscalar(y) - logistic_func(WT, x)
		xt = np.transpose(x)
		partialGradProduct = np.dot(xt,subtraction)
		sumOfVectors += partialGradProduct
	return sumOfVectors


#Function to check our accuracy and guesses on our final weights
def calcAccuracy(W,X,Y):
	#Counter of correct guesses overall
	counter = 0
	#Counter of where I guessed a participant was coming and they indeed were
	correctComing = 0
	#Counter of how many people I gussed were coming (regardless if they actually came or not)
	guessComing = 0
	#Counter of how many people I guessed were not coming (regardless if they actually didnt come or not)
	guessNotComing = 0
	#Counter of how many people I gussed were not comming AND they are actually not coming
	correctNotComing = 0

	#Get final error
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	WT = np.transpose(W)
	error = error_func(W,X,Y)
	print "Final Error is: " + str(error)

	#Check how right we were with these weights
	actual_coming, actual_not_coming = getStats(Y)
	#For each person, we check the results of what our logistic function gives
	for (xi,y) in zip(X,Y):
		result = logistic_func(WT,xi)
		#Sigmoid is above 0.5, we classify as YES/COMING
		if(result >= 0.5):
			Z.append(1)
			guessComing += 1
		else:
			Z.append(0)
			guessNotComing += 1
		#We said YES/COMING, AND, they are indeed coming
		if result >= 0.5 and y==1:
			correctComing += 1
		if result < 0.5 and y==0:
			correctNotComing += 1
		#Tabulate OVERALL number of CORRECT guesses
		if (result >= 0.5 and y == 1) or (result < 0.5 and y == 0):
			counter += 1

	comingRatio = (float(correctComing)/actual_coming)*100
	notComingRatio = (float(correctNotComing)/actual_not_coming)*100

	#Define success rate by number of CORRECT guesses, over number of TOTAL guesses
	print len(X)
	successrate = 100 * (float(counter)/float(len(X)))
	print "Out of: " + str(actual_coming) + " actually coming, " + "guessed " + str(guessComing) + " are coming, " + str(correctComing) + " are correct."
	print "Out of: " + str(actual_not_coming) + " actually not coming, " + "guessed " + str(guessNotComing) + " are not coming, " + str(correctNotComing) + " are correct."
	print "Accuracy rate of guessing a returning participant: " + str(comingRatio)
	print "Accuracy rate of guessing a non-returning participant: " + str(notComingRatio)
	print "Overall Accuracy Rate: " + str(successrate) + "%"
	plt.scatter(*zip(*training_error))
	plt.xlabel('Iterations')
	plt.ylabel('Error')
	plt.suptitle('Iterations vs Error during Gradient Descent in Logistic Regression')
	plt.show()
	print_predicted_attendance(Z,Y)


def main(X, Y, W):
	#Shuffle training data
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]

	W = gradient_desc(W,X,Y)
	calcAccuracy(W,X,Y)
	print W



def print_predicted_attendance(Z,Y):
	with open("Results.csv","w") as f:
	    wr = csv.writer(f,delimiter="\n")
	    wr.writerow(Z)

	with open("compareResults.csv","w") as f:
		wr = csv.writer(f,delimiter="\n")
		wr.writerow(Y)


def getStats(Y):
	numberOfYes = 0
	numberOfNo = 0
	for entry in Y:
		if entry == 1:
			numberOfYes += 1
		else:
			numberOfNo +=1
	return numberOfYes,numberOfNo

main(X,Y,W)

