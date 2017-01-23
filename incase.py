X2 = []
with open('parsedOutput.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
		X2.append(row)

Y2 = []
with open('result.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
		Y2.append(row)


X2 = X2[1:]
Y2 = Y2[1:]

X = []
Y = []


for entry in X2:
	entry = [float(i) for i in entry]
	entry.pop(0)
	X.append(entry)

for entry in Y2:
	entry = [float(i) for i in entry]
	entry.pop(0)
	Y.append(entry)


X = np.matrix(X)
Y = np.matrix(Y)
W = np.zeros(8)
W = np.matrix(W)
W = np.transpose(W)


X = X.astype(np.float)
Y = Y.astype(np.float)
W = W.astype(np.float)

Z = []














X = []
with open('parsedData.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X.append(row)

Y = []
with open('attendance.csv','rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        Y.append(row)

X = X[0:]
Y = Y[0:]

X = np.matrix(X)
Y = np.matrix(Y)
W = np.zeros(3)
W = np.matrix(W)
W = np.transpose(W)


X = X.astype(np.float)
Y = Y.astype(np.float)
W = W.astype(np.float)
Z = []