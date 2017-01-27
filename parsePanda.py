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
dataPredict = pd.DataFrame.from_csv("Project1_data.csv",infer_datetime_format=True)

data = data[0:]
#remove half marathon
data = data[data.Year != 2013]

#remove undefined sex
data = data[data.Sex != 'U']

#remove uneeded columns
data = data.drop('Name', 1)
dataPredict = dataPredict.drop('Name', 1)

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

sex_data_2017 = dataPredict['Sex'].map({'M':1, 'F': 0, 'U': 0})
sex_data_2017 = sex_data_2017.groupby(dataPredict.index).max()

#Aggregate ranks by ID
data['Rank'] = pd.to_numeric(data['Rank'])
avg_rank_data = data.Rank.groupby(data.index).mean()
avg_rank_data_2017 = data.Rank.groupby(data.index).mean()

dataPredict['Rank'] = pd.to_numeric(dataPredict['Rank'])
avg_rank_data_2017 = dataPredict.Rank.groupby(dataPredict.index).mean()
avg_rank_data_2017 = dataPredict.Rank.groupby(dataPredict.index).mean()

#Convert time to seconds
data['Time'] = data['Time'].map(time_convert)
avg_time_data = data.Time.groupby(data.index).mean()

dataPredict['Time'] = dataPredict['Time'].map(time_convert)
avg_time_data_2017 = dataPredict.Time.groupby(dataPredict.index).mean()

#Years Since Last Race
years_since_last = data.Year.groupby(data.index).max()
years_since_last = years_since_last.apply(lambda x: 2016-x)

years_since_last_2017 = dataPredict.Year.groupby(dataPredict.index).max()
years_since_last_2017 = years_since_last_2017.apply(lambda x: 2017-x)

#Years of participation
years_of_participation = data.Year.groupby(data.index).agg('count')

years_of_participation_2017 = dataPredict.Year.groupby(dataPredict.index).agg('count')


#Latest age
latest_age = data['Age Category'].groupby(data.index).max()

latest_age_2017 = dataPredict['Age Category'].groupby(dataPredict.index).max()


#Interaction Term b/w avgRank and years since last marathon
avg_rank_x_years_since_last = avg_rank_data.multiply(years_since_last)

avg_rank_x_years_since_last_2017 = avg_rank_data_2017.multiply(years_since_last_2017)


# #Attended 2015
# temp = data.copy()
# temp['AttendanceIn2015'] = 1
# temp['AttendanceIn2015'][temp['Year'] != 2015] = 0
# temp = temp.AttendanceIn2015.groupby(data.index).max()
# attended_in_2015 = temp.copy()

#Average pace
data['Pace'] = data['Pace'].map(time_convert_pace)
avg_pace_data = data.Pace.groupby(data.index).mean()


dataPredict['Pace'] = dataPredict['Pace'].map(time_convert_pace)
avg_pace_data_2017 = dataPredict.Pace.groupby(dataPredict.index).mean()

#concat into a single matrix
data = pd.concat([sex_data ,avg_rank_data, avg_time_data, years_since_last, years_of_participation, latest_age, results], axis=1)
data.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge', 'Attended2016'])

dataPredict = pd.concat([sex_data_2017 ,avg_rank_data_2017, avg_time_data_2017, years_since_last_2017, years_of_participation_2017, latest_age_2017], axis=1)
dataPredict.columns = (['Sex','AvgRank','AvgTime','YearsSinceLast','YearsOfParticipation','LatestAge'])

dataPredict.to_csv('predict2017.csv')


np.random.seed(94)
data = data.iloc[np.random.permutation(np.arange(len(data)))]


#2017 Setup
trainY2017 = data[data.columns[6]]
trainX2017 = data.copy()
trainX2017 = trainX2017.drop('Attended2016', 1)

trainX2017.to_csv('trainX2017.csv',header=True)
trainY2017.to_csv('trainY2017.csv',header=True)




#testX now contains 20% of my data
testX, temp1, temp2, temp3, temp4 = np.array_split(data,5)

trainX = pd.concat([temp1,temp4])
validationX = pd.concat([temp2,temp3])


testYCount = testX.groupby('Attended2016').count()
validateYCount = validationX.groupby('Attended2016').count()
trainYCount = trainX.groupby('Attended2016').count()

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


testY.columns=(['Attended2016'])
trainY.columns=(['Attended2016'])
validationY.columns=(['Attended2016'])

#CSVs for Training
trainX.to_csv('trainX.csv', header=True)
trainY.to_csv('trainY.csv', header=True)



#CSVs for Validation
validationX.to_csv('valX.csv',header=True)
validationY.to_csv('valY.csv',header=True)


#CSVs for Test
testX.to_csv('testX.csv',header=True)
testY.to_csv('testY.csv',header=True)

