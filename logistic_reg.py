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

#Test
Xtest = getCSVAndFormat('testX.csv')
Ytest = getCSVAndFormat('testY.csv')

#2017
X2017 = getCSVAndFormat('trainX2017.csv')
Y2017 = getCSVAndFormat('trainY2017.csv')
wholeSet = getCSVAndFormat('predict2017.csv')


#Weights
W = np.zeros(7)
W = np.matrix(W)
W = np.transpose(W)
W = W.astype(np.float)

Z = []
predictionsFor2017 = []


#Calculating sigmoid
def calculate_sigmoid(WT, x):
	x = np.transpose(x)
	WTx = np.dot(WT,x)
	WTx = np.asscalar(WTx)
	return 1/(1+np.power(np.e,-WTx))

#Calculating the error function in logistic regression
def error_func(W, X, Y):
	totalError = 0.0
	WT = np.transpose(W)
	#iterate through each "person/row" to sum the total error
	for(x,y) in zip(X,Y):
		sigmoid = calculate_sigmoid(WT,x)
		totalError += (np.asscalar(y) * math.log(sigmoid)) + ((1.0-np.asscalar(y)) * math.log(1.0 - sigmoid))
	return -totalError

#applying gradient descent on the derived error function to train weights
def gradient_desc(W, X, Y, alpha=.0001, tolerance= 0.5):
	X = normalize_add_intercept(X)
	error = error_func(W, X, Y)
	delta = 1
	i=0
	#We continue iteration until our error is below a tolerance threshold. We define delta as the OldError - NewError. When delta is sufficiently small, stop iteration.
	while(tolerance < delta):
		error_prev = error
		W = W + (alpha * (gradient(W, X, Y)))
		error = error_func(W, X, Y)
		delta = error_prev - error
		print "Iteration: " + str(i) + " Delta: " + str(delta) + " Error: " + str(error)
		i+=1
	return W

#Calculate the term we will be multiplying our learning rate alpha by.
def gradient(W, X, Y):
	sumOfVectors = 0.0
	WT = np.transpose(W)
	#Need to iterate through each "person/row"
	for(x,y) in zip(X,Y):
		subtraction = np.asscalar(y) - calculate_sigmoid(WT, x)
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

	X = normalize_add_intercept(X)
	# #Get final error
	# WT = np.transpose(W)
	# error = error_func(W,X,Y)
	# print "Final Error is: " + str(error)

	#Check how right we were with these weights
	actual_coming, actual_not_coming = getStats(Y)
	#For each person, we check the results of what our logistic function gives
	WT = np.transpose(W)
	for (xi,y) in zip(X,Y):
		result = calculate_sigmoid(WT,xi)
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

	accuracy = (float(counter)/float(len(X)))

	precision = float(correctComing)/(guessComing)

	recall = float(correctComing)/(actual_coming)

	falsePositiveRate = float((guessComing-correctComing))/((guessComing-correctComing) + correctNotComing)

	if recall != 0:
		F1Measure = 2*((precision*recall)/(precision+recall))


	print "Accuracy: " + str(accuracy)
	print "Precision: " + str(precision)
	print "Recall: " + str(recall)
	print "False Positive  Rate: " + str(falsePositiveRate)
	print "F1 Measure: " + str(F1Measure)


	print "Out of: " + str(actual_coming) + " actually coming, " + "guessed " + str(guessComing) + " are coming, " + str(correctComing) + " are correct."
	print "Out of: " + str(actual_not_coming) + " actually not coming, " + "guessed " + str(guessNotComing) + " are not coming, " + str(correctNotComing) + " are correct."
	print "Accuracy rate of guessing a returning participant: " + str(comingRatio)
	print "Accuracy rate of guessing a non-returning participant: " + str(notComingRatio)
	# plt.scatter(*zip(*training_error))
	# plt.xlabel('Iterations')
	# plt.ylabel('Error')
	# plt.suptitle('Iterations vs Error during Gradient Descent in Logistic Regression')
	# plt.show()
	print_predicted_attendance(Z,Y)


def main(X, Y, W):
	#Shuffle training data
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]

	#TWO FOLD
	W1 = gradient_desc(W,X,Y)
	calcAccuracy(W1,Xval,Yval)
	W2 = gradient_desc(W,Xval,Yval)
	calcAccuracy(W2,X,Y)

	print "=================ON TEST SET=========================================="

	calcAccuracy(W2,Xtest,Ytest)
	# #2017
	# W = gradient_desc(W,X,Y)
	# get2017Predictions(W,wholeSet)


	print W1
	print W2



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



def get2017Predictions(W,X):
	X = normalize_add_intercept(X)
	WT = np.transpose(W)
	for xi in X:
		result = calculate_sigmoid(WT,xi)
		#Sigmoid is above 0.5, we classify as YES/COMING
		if(result >= 0.5):
			predictionsFor2017.append(1)
		else:
			predictionsFor2017.append(0)

	with open("2017PredictionsFinal.csv","w") as f:
		wr = csv.writer(f,delimiter="\n")
		wr.writerow(predictionsFor2017)




def normalize_add_intercept(X):
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	X = np.c_[np.ones(X.shape[0]),X]
	return X


main(X,Y,W)

