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



def calculateProbabilityDensity(x, mean, standardDev):
	rightSide = math.exp(- (math.pow(x - mean, 2) / (2 * math.pow(standardDev , 2)) ))
	leftSide =  1 / (math.sqrt(2 * math.pi) * standardDev)
	return leftSide * rightSide

def calculateClassProbabilities(featureMap, inputRow):
	probabilities = {}
	for classValue, classFeatures in featureMap.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classFeatures)):
			mean, standardDev = classFeatures[i]
			x = inputRow[i]
			probabilities[classValue] *= calculateProbabilityDensity(x, mean, standardDev)
	return probabilities

def predictForFeatures(featureMap, inputRow):
	probabilities = calculateClassProbabilities(featureMap, inputRow)
	bestLabel = 1
	bestProbability = probabilities[1]
	if (probabilities[0] > bestProbability):
		bestLabel = 0
	return bestLabel

def getPredictions(featureMap, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predictForFeatures(featureMap, testSet[i])
		predictions.append(result)
	return predictions

#for testing purposes only
def getTotalAccuracy(testSet, predictions):
	correctPredictions = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correctPredictions += 1
	return (correctPredictions / float(len(testSet))) * 100.0

#for testing purposes only
def getAttendanceAccuracy(testSet, predictions):
	predictedAttendance = 0
	correctPredictions = 0
	for x in range(len(testSet)):
		if predictions[x] == 1:
			predictedAttendance += 1
			if testSet[x][-1] == 1:
				correctPredictions += 0
	print ("correct predictions: %d" % correctPredictions) 
	print ("total attendance predictions: %d" % predictedAttendance)

	return (correctPredictions/float(predictedAttendance)) * 100.0

def readAndConcatFeaturesToResult(file1, file2):
	features = pd.read_csv(file1, header = 0)
	features.drop(["Id"], inplace = True, axis = 1)
	Y = pd.read_csv(file2, header = 0)
	Y.drop(["Id"], inplace = True, axis = 1)
	features["Attended2016"] = Y["Attended2016"]
	return features

def main():
	XFileName = "trainX2017.csv"
	YFileName = "trainY2017.csv"
	features = readAndConcatFeaturesToResult(XFileName, YFileName)
	#concat 

	#separate features into two classes
	yesFeatures = features[(features.Attended2016 == 1)]
	noFeatures = features[(features.Attended2016 == 0)]

	#calculate mean and st.dev for each class
	yesFeaturesNormalized = [(yesFeatures[column].mean(), yesFeatures[column].std()) for column in yesFeatures]
	noFeaturesNormalized = [(noFeatures[column].mean(), noFeatures[column].std()) for column in noFeatures]

	print yesFeaturesNormalized
	print noFeaturesNormalized

	classToFeatures = {}
	classToFeatures[1] = yesFeaturesNormalized[0 : -1]
	classToFeatures[0] = noFeaturesNormalized[0 : -1]

	print classToFeatures
	# validation = readAndConcatFeaturesToResult(XValFile, YValFile)
	validation = pd.read_csv("predict2017.csv", header = 0)
	predictions = getPredictions(classToFeatures, validation.as_matrix())
	with open("2017BayesPredictions.csv", "w") as bayesOutputFile:
		wr = csv.writer(bayesOutputFile,delimiter = "\n")
		wr.writerow(predictions)

	# print('Total Accuracy: {0}%').format(totalAccuracy)
	# print('Attendance Accuracy: {0}%').format(attendanceAccuracy)






if __name__ == "__main__":
	main()