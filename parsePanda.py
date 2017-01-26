import pandas as pd
import sklearn.utils 
import numpy as np

def is2016(x):
	if x == 2016:
		return 1
	else:
		return 0

def attended2015(x):
	if x == 2015:
		return 1
	else:
		return 0


def time_convert(x):
    timeStamp = x.split(':')
    return 3600*int(timeStamp[0])+60*int(timeStamp[1])+int(timeStamp[2])

def time_convert_pace(x):
	timeStamp = x.split(':')
	return int(timeStamp[0])*60 + int(timeStamp[1])


data = pd.DataFrame.from_csv("Project1_data.csv",infer_datetime_format=True)

data = data[0:]
#remove half marathon
data = data[data.Year != 2013]

#remove undefined sex
data = data[data.Sex != 'U']

#remove uneeded columns
data = data.drop('Name', 1)

#Get my Y matrix (did they come in 2016)
Y_Ids = data.Year.groupby(data.index).mean()
Y_Ids = Y_Ids[Y_Ids < 2016]
results = data.join(Y_Ids, how='inner', lsuffix='_other')
Y_max = data.Year.groupby(data.index).max()
results = results.join(Y_max, how='inner', lsuffix='_other2')
results = results.Year.groupby(results.index).mean()
results = results.apply(is2016)
results.columns = (['Attended'])


#remove 2016
data = data[data.Year != 2016]


#We want the following categories:
#sex, average rank, average Time, and years since last race


#Map the sex to binary
sex_data = data['Sex'].map({'M': 1, 'F': 0})
sex_data = sex_data.groupby(data.index).max()

#Aggregate ranks by ID
data['Rank'] = pd.to_numeric(data['Rank'])
avg_rank_data = data.Rank.groupby(data.index).mean()

#Convert time to seconds
data['Time'] = data['Time'].map(time_convert)
avg_time_data = data.Time.groupby(data.index).mean()

#Years Since Last Race
years_since_last = data.Year.groupby(data.index).max()
years_since_last = years_since_last.apply(lambda x: 2016-x)

#Years of participation
years_of_participation = data.Year.groupby(data.index).agg('count')

#Latest age
latest_age = data['Age Category'].groupby(data.index).max()

# #Attended 2015
# temp = data.copy()
# temp['AttendanceIn2015'] = 1
# temp['AttendanceIn2015'][temp['Year'] != 2015] = 0
# temp = temp.AttendanceIn2015.groupby(data.index).max()
# attended_in_2015 = temp.copy()


#Average pace
data['Pace'] = data['Pace'].map(time_convert_pace)
avg_pace_data = data.Pace.groupby(data.index).mean()

#concat into a single matrix
data = pd.concat([sex_data ,avg_rank_data, avg_time_data, years_since_last, years_of_participation, latest_age, results], axis=1)
data.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge', 'Attended2016'])


data = sklearn.utils.shuffle(data,random_state=0)


# np.random.seed(97)
# data = data.iloc[np.random.permutation(np.arange(len(data)))]

#testX now contains 20% of my data
testX, temp1, temp2, temp3, temp4 = np.array_split(data,5)

trainX = pd.concat([temp1,temp4,temp3])
#validationX = pd.concat([temp2,temp3])
validationX = temp2

# #data now has 80% of my data
# data = pd.concat([temp1,temp2,temp3,temp4])

# #trainX and validationX have a 50/50 split
# validationX, trainX = np.array_split(data,2)


#extract trainY
trainY = trainX[trainX.columns[6]]
trainY.columns = (['Attended2016'])


validationY = validationX[validationX.columns[6]]
validationY.columns = (['Attended2016'])

testY = testX[testX.columns[6]]
testY.columns = (['Attended2016'])

del trainX['Attended2016']
del validationX['Attended2016']
del testX['Attended2016']


# #Split train,test,validation
# trainX,testX = train_test_split(data,test_size = 0.2,random_state=94)
# trainX,validationX = train_test_split(trainX, test_size = 0.5,random_state=94)

# #Split train,test,validation
# trainY,testY = train_test_split(results,test_size = 0.2,random_state=94)
# # trainY,validationY = train_test_split(trainY, test_size = 0.5,random_state=94)

# trainX.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge', 'AttendedIn2015'])
# validationX.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge', 'AttendedIn2015'])
# testX.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge', 'AttendedIn2015'])

#CSVs for Training
trainX.to_csv('trainX.csv', header=True)
trainY.to_csv('trainY.csv', header=True)



#CSVs for Validation
validationX.to_csv('valX.csv',header=True)
validationY.to_csv('valY.csv',header=True)


#CSVs for Test
testX.to_csv('testX.csv',header=True)
testY.to_csv('testY.csv',header=True)



# def shuffle_custom(data):
# 	length = len(data)
# 	end = int(length * 0.8)

# 	parsedData = np.array(parsedData)

# 	mergedData = np.c_[parsedData, parsedResult]

# 	np.random.seed(0)
# 	randomize = np.arange(len(mergedData))
# 	np.random.shuffle(randomize)
# 	mergedData = mergedData[randomize]

# 	print mergedData

# 	(row, col) = mergedData.shape

# 	parsedResult = mergedData[:, col - 1]
# 	parsedData = np.delete(mergedData, [col - 1], axis=1)

# 	train_validate_data = parsedData[0:end]	
# 	train_validate_result = parsedResult[0:end]

# 	test_data = parsedData[end + 1 :]
# 	test_result = parsedResult[end + 1 :]

