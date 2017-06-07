print("regularizing data")
max_values = [None]*numVariables
min_values = [None]*numVariables
for row in data:
    for i in range(numVariables):
        if max_values[i] == None or row[i] > max_values[i]:
            max_values[i] = row[i]
        if min_values[i] == None or row[i] < min_values[i]:
            min_values[i] = row[i]

regularization = [-1,1] #min and max values after regularization

for row in data:
    for i in range(numVariables):
        row[i] = ((row[i] - min_values[i]) / (max_values[i] - min_values[i])) * (regularization[1]- regularization[0]) + regularization[0]i
