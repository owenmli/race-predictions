import pandas as pd

def is2016(x):
	if x == 2016:
		return 1
	else:
		return 0


def time_convert(x):
    timeStamp = x.split(':')
    return 3600*int(timeStamp[0])+60*int(timeStamp[1])+int(timeStamp[2])


data = pd.DataFrame.from_csv("Project1_data.csv",infer_datetime_format=True)
data = data[0:]
#remove half marathon
data = data[data.Year != 2013]
#remove undefined sex
data = data[data.Sex != 'U']
#remove uneeded columns
data = data.drop('Name', 1)
data = data.drop('Age Category', 1)
data = data.drop('Pace', 1)

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
avg_time_data = avg_time_data.apply(lambda x: x/2)

#Years Since Last Race
years_since_last = data.Year.groupby(data.index).max()
years_since_last = years_since_last.apply(lambda x: 2016-x)

#concat into a single matrix
data = pd.concat([sex_data, avg_rank_data, avg_time_data, years_since_last], axis=1)
data = avg_time_data

data.to_csv('parsedData.csv')
results.to_csv('2016Attendance.csv', header=True)


