import csv

#PARAMETERS:
#SEX, AVG RANK, AVG TIME, YEAR SINCE LAST MARATHON
#1= MALE, 2=FEMALE


data = []
with open('Project1_data.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

i=0
data = data[1:]
X = []
Y= []
individualParsedData = []

while i < len(data):
	participantID = data[i][0]
	partcipantData = []

	start = i

	for j in range(start, len(data)):
		if data[j][0] == participantID:
			partcipantData.append(data[j])
			i = j+1
		else:
			i = j
			break

	sex = 0
	avgRank = 0
	avgTime = 0
	yearsSinceLast = 0

	#MALE = 1 FEMALE = 0
	if partcipantData[0][3] == "M":
		sex = 1
	else:
		sex = 0

	rankSum = 0
	timeSum = 0
	lastMarathonYear = 0
	for row in partcipantData:
		rankSum = rankSum + int(row[4])
		parsedTime = row[5].split(':')
		timeSum = timeSum + int(parsedTime[0]) * 3600 + int(parsedTime[1]) * 60 + int(parsedTime[2])
		if int(row[7]) > lastMarathonYear:
			lastMarathonYear = int(row[7])

	avgRank = rankSum/len(partcipantData)
	avgTime = timeSum/len(partcipantData)

	yearsSinceLast = 2017-lastMarathonYear

	individualParsedData = [sex, avgRank, avgTime,yearsSinceLast]
	X.append(individualParsedData)

with open("parsedData.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(X)

	# 1 = ATTENDED 2016
	if yearsSinceLast == 1:
		Y.append(1)
	else:
		Y.append(0)


with open("attendance.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(Y)
