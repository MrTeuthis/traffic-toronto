import pickle
import numpy as np

print("loading pickle dump")
data = np.asarray(pickle.load(open("complete-dump.pkl", "rb")))
numVariables = len(data[0])
np.random.shuffle(data)

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
        row[i] = ((row[i] - min_values[i]) / (max_values[i] - min_values[i])) * (regularization[1]- regularization[0]) + regularization[0]

print("separating inputs and outputs")
inputs = data[:,:8]
outputs = data[:,8:]

print("grouping data into training, validation, and testing")
breakdown = [0.6, 0.2, 0.2] #breakdown of data into training, validation, and testing groups
slice1 = int(breakdown[0]*len(data))
slice2 = slice1 + int(breakdown[1]*len(data))

training_inputs = inputs[:slice1]
validation_inputs = inputs[slice1:slice2]
testing_inputs = inputs[slice2:]
training_outputs = outputs[:slice1]
validation_outputs = outputs[slice1:slice2]
testing_outputs = outputs[slice2:]


print("dumping regularization parameters, training, validation, and test data to pickle")
pickle.dump(max_values, open("max-values.pkl", "wb"))
pickle.dump(min_values, open("min-values.pkl", "wb"))
pickle.dump(training_inputs, open("training-inputs.pkl", "wb"))
pickle.dump(validation_inputs, open("validation-inputs.pkl", "wb"))
pickle.dump(testing_inputs, open("testing-inputs.pkl", "wb"))
pickle.dump(training_outputs, open("training-outputs.pkl", "wb"))
pickle.dump(validation_outputs, open("validation-outputs.pkl", "wb"))
pickle.dump(testing_outputs, open("testing-outputs.pkl", "wb"))

