import tensorflow as tf
import random as rand
import numpy as np
import pickle
import random
import argparse
import pprint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.animation import FuncAnimation
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train the neural net", action="store_true")
parser.add_argument("--feedforward", help="run the neural net on data", action="store_true")
parser.add_argument("--input", help="the input of the neural net (a .pkl file)", action="store")
parser.add_argument("--output", help="the output of the neural net (a .pkl file)", action="store")
parser.add_argument("--weather", help="make weather gifs", action="store_true")
parser.add_argument("--time", help="make week chart", action="store_true")
args=parser.parse_args()

if (args.train and (args.feedforward or args.input or args.output)) or ((args.feedforward or args.input or args.output) and not (args.feedforward and args.input and args.output)):
    print("invalid arguments")
    exit(1)

# To suppress debug output, run python or python3 with the -O option:
#$ python -O tfmat.py

# PLEASE UPDATE THESE AS REQUIRED before trying to run tyvm

# Number of features in the input.
# This has *nothing to do* with the number of training examples.
NUM_IN = 8

# Number of nodes in the hidden layer.
NUM_HIDDEN = 5

# Number of features in the output. 
NUM_OUT = 2

# The learning rate of the neural network. 
LEARNING_RATE = 0.3

def stringify_tensors(labels: list, tensors: list, sep='\n'):
    return sep.join([str(label) + ': ' + str(tensor) for label, tensor in zip(labels, tensors)])

def visualize_tensors(labels: list, tensors: list, start='', end='\n'):
    return start + stringify_tensors(labels, tensors) + end

with tf.name_scope('in'):
    x = tf.placeholder(tf.float32, [None, NUM_IN], name='x') #inputs

y_ = tf.placeholder(tf.float32, [None, NUM_OUT], name='true_ys')

if args.train:
    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            W1 = tf.Variable(tf.random_uniform([NUM_IN, NUM_HIDDEN], -1.0)) #weights of hiddens
        with tf.name_scope('biases'):
            b1 = tf.Variable(tf.random_uniform([NUM_HIDDEN], -1.0)) #biases
        with tf.name_scope('preacts'):
            pre1 = tf.matmul(x, W1) + b1 #pre-activations of hiddens
        with tf.name_scope('acts'):
            H1 = tf.sigmoid(pre1) #activations of hiddens

    with tf.name_scope('out'):
        with tf.name_scope('weights'):
            W2 = tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_OUT], -1.0)) #weights of outs
        with tf.name_scope('biases'):
            b2 = tf.Variable(tf.random_uniform([NUM_OUT], -1.0)) #biases of outs
        with tf.name_scope('preacts'):        
            pre2 = tf.matmul(H1, W2) + b2 #pre-activations of outs
        with tf.name_scope('acts'):
            y = tf.identity(pre2) #activations of outs

    with tf.name_scope('cost'):
        mse_cost = tf.losses.mean_squared_error(y_, y)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(mse_cost)

    #get data from pickle
    training_inputs = pickle.load(open("data/training-inputs.pkl","rb"))
    training_outputs = pickle.load(open("data/training-outputs.pkl","rb"))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        iteration = 1
        while True:
            try:
                batch_xs, batch_ys = training_inputs, training_outputs
                _, cost, yr, W2r, b2r, H1r, W1r, b1r, xr = sess.run(
                    (train_step, mse_cost, y, W2, b2, H1, W1, b1, x),
                    feed_dict={x: batch_xs, y_: batch_ys}
                    )
                print("iteration " + str(iteration))
                iteration += 1
                print(visualize_tensors(["COST"],[cost]))
            except KeyboardInterrupt:
                break
    
    #dump weights to pickle
    print("\ndumping weights to pickle")
    pickle.dump(W1r, open("weights/W1r.pkl","wb"))
    pickle.dump(W2r, open("weights/W2r.pkl","wb"))
    pickle.dump(b1r, open("weights/b1r.pkl","wb"))
    pickle.dump(b2r, open("weights/b2r.pkl","wb"))

if args.feedforward:
    data_input = pickle.load(open(args.input,"rb"))
    data_output = pickle.load(open(args.output,"rb"))
    W1r = pickle.load(open("weights/W1r.pkl","rb"))
    W2r = pickle.load(open("weights/W2r.pkl","rb"))
    b1r = pickle.load(open("weights/b1r.pkl","rb"))
    b2r = pickle.load(open("weights/b2r.pkl","rb"))

    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            W1 = tf.Variable(W1r) #weights of hiddens
        with tf.name_scope('biases'):
            b1 = tf.Variable(b1r) #biases
        with tf.name_scope('preacts'):
            pre1 = tf.matmul(x, W1) + b1 #pre-activations of hiddens
        with tf.name_scope('acts'):
            H1 = tf.sigmoid(pre1) #activations of hiddens

    with tf.name_scope('out'):
        with tf.name_scope('weights'):
            W2 = tf.Variable(W2r) #weights of outs
        with tf.name_scope('biases'):
            b2 = tf.Variable(b2r) #biases of outs
        with tf.name_scope('preacts'):        
            pre2 = tf.matmul(H1, W2) + b2 #pre-activations of outs
        with tf.name_scope('acts'):
            y = tf.identity(pre2) #activations of outs

    with tf.name_scope('cost'):
        mse_cost = tf.losses.mean_squared_error(y_, y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        y_out, cost = sess.run((y, mse_cost), feed_dict={x:data_input, y_:data_output})
        print(visualize_tensors(["cost"], [cost]))

    print("regularized maximums")
    print(np.amax(y_out,0))

    print("regularized minimums")
    print(np.amin(y_out,0))

    print("de-regularizing data")
    max_values = pickle.load(open("data/max-values.pkl","rb"))[8:10]
    min_values = pickle.load(open("data/min-values.pkl","rb"))[8:10]
    regularization = [-1,1]

    for i in range(len(y_out)):
        for j in range(len(y_out[0])):
            y_out[i][j] = (y_out[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j] - min_values[j])
            data_output[i][j] = (data_output[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j] - min_values[j])

    print ("previous maximums")
    print(max_values)

    print("maximum outputs")
    print(np.amax(y_out,0))

    print("previous minimums")
    print(min_values)

    print("minimum outputs")
    print(np.amin(y_out,0))

    difference = np.absolute(y_out - data_output)
    print("average absolute difference")
    print(np.average(difference, 0))

    print("standard deviation of each output")
    print(np.std(y_out - data_output,0))

    print("max difference")
    print(np.ndarray.max(difference, 0))
    
    numBins = 100

    print("histogram for differences of relative time")
    histogram = np.histogram(difference[:,0],bins=numBins)
    for i in range(numBins):
        print(str(histogram[1][i]) + " " + str(histogram[0][i]))

    print("histogram for differences of volume")
    histogram = np.histogram(difference[:,1],bins=numBins)
    for i in range(numBins):
        print(str(histogram[1][i]) + " " + str(histogram[0][i]))

if args.weather:
    W1r = pickle.load(open("weights/W1r.pkl","rb"))
    W2r = pickle.load(open("weights/W2r.pkl","rb"))
    b1r = pickle.load(open("weights/b1r.pkl","rb"))
    b2r = pickle.load(open("weights/b2r.pkl","rb"))
    

    data_list = []
    numSteps = 30
    temperature = [0] * (numSteps + 1)
    dewPoint = [0] * (numSteps + 1)
    humidity = [0] * (numSteps + 1)
    for i in range(numSteps+1):
        temperature[i] = i/numSteps * 2 - 1
        for j in range(numSteps+1):
            dewPoint[j] = j/numSteps * 2 - 1
            for k in range(numSteps+1):
                humidity[k] = k/numSteps * 2 - 1
                data_list.append([0,0,0.5,0,0,temperature[i], dewPoint[j], humidity[k]])
    data_input = np.asarray(data_list)

    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            W1 = tf.Variable(W1r) #weights of hiddens
        with tf.name_scope('biases'):
            b1 = tf.Variable(b1r) #biases
        with tf.name_scope('preacts'):
            pre1 = tf.matmul(x, W1) + b1 #pre-activations of hiddens
        with tf.name_scope('acts'):
            H1 = tf.sigmoid(pre1) #activations of hiddens

    with tf.name_scope('out'):
        with tf.name_scope('weights'):
            W2 = tf.Variable(W2r) #weights of outs
        with tf.name_scope('biases'):
            b2 = tf.Variable(b2r) #biases of outs
        with tf.name_scope('preacts'):        
            pre2 = tf.matmul(H1, W2) + b2 #pre-activations of outs
        with tf.name_scope('acts'):
            y = tf.identity(pre2) #activations of outs

    with tf.name_scope('cost'):
        mse_cost = tf.losses.mean_squared_error(y_, y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        y_out = sess.run(y, feed_dict={x:data_input})
    
    print("de-regularizing data")
    max_values = pickle.load(open("data/max-values.pkl","rb"))
    min_values = pickle.load(open("data/min-values.pkl","rb"))
    regularization = [-1,1]

    for i in range(len(y_out)):
        for j in range(len(y_out[0])):
            y_out[i][j] = (y_out[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j+8] - min_values[j+8]) + min_values[j+8]
    for i in range(len(data_input)):
        for j in range(len(data_input[0])):
            data_input[i][j] = (data_input[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j] - min_values[j]) + min_values[j]
    
    for i in range(len(temperature)):
        temperature[i] = (temperature[i] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[5] - min_values[5]) + min_values[5]
        dewPoint[i] = (dewPoint[i] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[6] - min_values[6]) + min_values[6]
        humidity[i] = (humidity[i] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[7] - min_values[7]) + min_values[7]

    index = 0
    weather_outputs = np.zeros((numSteps+1, numSteps+1, numSteps+1, 2))
    for i in range(numSteps+1):
        for j in range(numSteps+1):
            for k in range(numSteps+1):
                weather_outputs[i][j][k] = y_out[index]
                index += 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(dewPoint, humidity)
    ax.set_zlim(np.amin(y_out, 0)[1], np.amax(y_out, 0)[1]) #change index for time/volume
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("Dew Point")
    ax.set_ylabel("Relative Humidity")
    ax.set_zlabel("Traffic Volume")
    ttl = ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    
    def update(i):
        Z = weather_outputs[i,:,:,1]    #change index for time/volume
        surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)
        ttl.set_text("temperature: " + str(round(temperature[i])))
        return surf, ttl
    
    if __name__ == '__main__':
        anim = FuncAnimation(fig, update, frames=np.arange(0, numSteps+1), interval=300)
        anim.save('weather.gif', dpi=80, writer='imagegick')

if args.time:
    W1r = pickle.load(open("weights/W1r.pkl","rb"))
    W2r = pickle.load(open("weights/W2r.pkl","rb"))
    b1r = pickle.load(open("weights/b1r.pkl","rb"))
    b2r = pickle.load(open("weights/b2r.pkl","rb"))
    
    data_list = []
    days = [0]*7
    time_intervals = (24*60)//60    #the last number is the time in minutes between data points
    times = [0]*time_intervals
    for i in range(len(days)):
        days[i] = (i/6.0)*2.0 - 1
        for j in range(len(times)):
            times[j] = (j/time_intervals)*2.0 - 1
            data_list.append([0,days[i],times[j], 0, 0, 0, 0, 0])

    data_input = np.asarray(data_list)

    with tf.name_scope('hidden1'):
        with tf.name_scope('weights'):
            W1 = tf.Variable(W1r) #weights of hiddens
        with tf.name_scope('biases'):
            b1 = tf.Variable(b1r) #biases
        with tf.name_scope('preacts'):
            pre1 = tf.matmul(x, W1) + b1 #pre-activations of hiddens
        with tf.name_scope('acts'):
            H1 = tf.sigmoid(pre1) #activations of hiddens

    with tf.name_scope('out'):
        with tf.name_scope('weights'):
            W2 = tf.Variable(W2r) #weights of outs
        with tf.name_scope('biases'):
            b2 = tf.Variable(b2r) #biases of outs
        with tf.name_scope('preacts'):        
            pre2 = tf.matmul(H1, W2) + b2 #pre-activations of outs
        with tf.name_scope('acts'):
            y = tf.identity(pre2) #activations of outs

    with tf.name_scope('cost'):
        mse_cost = tf.losses.mean_squared_error(y_, y)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        y_out = sess.run(y, feed_dict={x:data_input})
    
    print(data_input)

    print("deregularizing data")
    max_values = pickle.load(open("data/max-values.pkl","rb"))
    min_values = pickle.load(open("data/min-values.pkl","rb"))
    regularization = [-1,1]

    for i in range(len(y_out)):
        for j in range(len(y_out[0])):
            y_out[i][j] = (y_out[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j+8] - min_values[j+8]) + min_values[j+8]
    for i in range(len(data_input)):
        for j in range(len(data_input[0])):
            data_input[i][j] = (data_input[i][j] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[j] - min_values[j]) + min_values[j]

    for i in range(len(days)):
        days[i] = (days[i] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[1] - min_values[1]) + min_values[1]
    for i in range(len(times)):
        times[i] = (times[i] - regularization[0]) / (regularization[1] - regularization[0]) * (max_values[2] - min_values[2]) + min_values[2]

    x_axis_days = []
    for i in range(len(days)):
        for j in range(len(times)):
            x_axis_days.append(days[i] + times[j])
    
    for row in data_input:
        print(row)
    
    x = np.asarray(x_axis_days)
    volume = y_out[:,1]
    relative_time = y_out[:,0]

    plt.plot(x, relative_time, "ro")
    plt.show()

