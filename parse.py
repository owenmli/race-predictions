import csv

#PARAMETERS:
#SEX, AVG RANK, AVG TIME, YEARS SINCE LAST MARATHON
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
testSet = []
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

	nOfRaces = len(partcipantData)
	yearOfRace = int(partcipantData[0][7])

	#print partcipantData[0][1] + '   ' + str(nOfRaces)

	if nOfRaces == 1 and yearOfRace == 2016:
		print "SKIP PERSON: " + partcipantData[0][1]
		continue

	sex = 0
	avgRank = 0
	avgTime = 0
	yearsSinceLast = 0
	intercept = 1

	rankSum = 0
	timeSum = 0
	lastMarathonYear = 0
	cameIn2016 = 0


	#MALE = 1 FEMALE = 0
	if partcipantData[0][3] == "M":
		sex = 1
	else:
		sex = 0

	iparticipantData = iter(partcipantData)

	skipCount = 0
	print len(partcipantData)
	for row in iparticipantData:
		if int(row[7]) == 2016:
			print "SKIP 2016 RECORD FOR: " + row[1] + " ========= " + row[4]
			cameIn2016 = 1
			skipCount += 1
			iparticipantData.next	
			continue
		print "Current person: " + row[1] + " ================== " + row[4]
		rankSum = rankSum + int(row[4])
		parsedTime = row[5].split(':')
		timeSum = timeSum + int(parsedTime[0]) * 3600 + int(parsedTime[1]) * 60 + int(parsedTime[2])
		if int(row[7]) > lastMarathonYear:
			lastMarathonYear = int(row[7])

	print skipCount
	print len(partcipantData) - skipCount
	print rankSum

	avgRank = rankSum/(len(partcipantData) - skipCount)
	avgTime = timeSum/(len(partcipantData) - skipCount)

	yearsSinceLast = 2017-lastMarathonYear


	individualParsedData = [sex, avgRank, avgTime,yearsSinceLast]

	X.append(individualParsedData)

	# 1 = ATTENDED 2016
	if cameIn2016 == 1:
		Y.append(1)
	else:
		Y.append(0)



with open("parsedData.csv", "wb") as f:
	writer = csv.writer(f)
	writer.writerows(X)


with open("attendance.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(Y)
