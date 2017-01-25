import numpy as np
import csv
import math
import random
import argparse
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Steps for Naive Bayes:

1. Import data into X and Y

2. Shuffle data sets X and Y where X are interesting features and Y is attendance

3. Calculate mean values for each attribute X for each Y

4. Calculate standard deviation for each attribute X for each Y

5. Calculate Probability of each attribute X

6. Calculate Probability of X belonging to each class

7. Predict after making these estimations
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, input):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = input[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, input):
	probabilities = calculateClassProbabilities(summaries, input)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	classFile = "trainY.csv"
	featuresFile = "trainX.csv"

	featuresFrame = pd.read_csv(featuresFile, header=0)
	featuresFrame.drop(["Id"],inplace=True,axis=1)
	classFrame = pd.read_csv(classFile, header=0)
	classFrame.drop(["Id"],inplace=True,axis=1)
	featuresFrame["attendance"] = classFrame["Year"]
	featuresFrame = featuresFrame.sample(frac=1).reset_index(drop=True)

	testFrame = featuresFrame[3*len(featuresFrame)/4 + 1: ]
	featuresFrame = featuresFrame[0: 3*len(featuresFrame)/4]

	#separate into two classes
	attendedFrame = featuresFrame[(featuresFrame.attendance == 1)]
	unattendedFrame = featuresFrame[(featuresFrame.attendance == 0)]

	#calculate mean and st.dev for each 
	meanAndStDevAttended = [(attendedFrame[column].mean(), attendedFrame[column].std()) for column in attendedFrame]
	meanAndStDevUnAttended = [(unattendedFrame[column].mean(), unattendedFrame[column].std()) for column in unattendedFrame]

	summarizedValues = {}
	summarizedValues[meanAndStDevAttended[-1][0]] = meanAndStDevAttended[0:-1]
	summarizedValues[meanAndStDevUnAttended[-1][0]] = meanAndStDevUnAttended[0:-1]
	npTestMatrix = testFrame.as_matrix()

	predictions = getPredictions(summarizedValues, npTestMatrix)
	accuracy = getAccuracy(npTestMatrix, predictions)

	print('Accuracy: {0}%').format(accuracy)


	return attendedFrame, unattendedFrame



if __name__ == "__main__":
	main()