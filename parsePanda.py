import pandas as pd
from sklearn.cross_validation import train_test_split

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

#Attended 2015
temp = data.copy()
temp['AttendanceIn2015'] = 1
temp['AttendanceIn2015'][temp['Year'] != 2015] = 0
temp = temp.AttendanceIn2015.groupby(data.index).max()
attended_in_2015 = temp.copy()


#Average pace
data['Pace'] = data['Pace'].map(time_convert_pace)
avg_pace_data = data.Pace.groupby(data.index).mean()

#concat into a single matrix
data = pd.concat([avg_rank_data,years_since_last, years_of_participation, latest_age, attended_in_2015], axis=1)

#Split train,test,validation
trainX,testX = train_test_split(data,test_size = 0.2,random_state=94)
trainX,validationX = train_test_split(trainX, test_size = 0.5,random_state=94)

#Split train,test,validation
trainY,testY = train_test_split(results,test_size = 0.2,random_state=94)
trainY,validationY = train_test_split(trainY, test_size = 0.5,random_state=94)

trainX.columns = (['AvgRank','YearsSinceLast','YearsOfParticipation','LatestAge', 'AttendedIn2015'])
trainY.columns = ['Attendance']

#CSVs for Training
trainX.to_csv('trainX.csv', header=True)
trainY.to_csv('trainY.csv', header=True)



#CSVs for Validation
validationX.to_csv('valX.csv',header=True)
validationY.to_csv('valY.csv',header=True)


#CSVs for Test
testX.to_csv('testX.csv',header=True)
testY.to_csv('testY.csv',header=True)



