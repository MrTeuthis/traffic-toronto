import tensorflow as tf
import random as rand
import numpy as np
import pickle
import random

# To suppress debug output, run python or python3 with the -O option:
#$ python -O tfmat.py

# PLEASE UPDATE THESE AS REQUIRED before trying to run tyvm

# Number of features in the input.
# This has *nothing to do* with the number of training examples.
NUM_IN = 2

# Number of nodes in the hidden layer.
NUM_HIDDEN = 5

# Number of features in the output. 
NUM_OUT = 2

# The learning rate of the neural network. 
LEARNING_RATE = 0.3

# The location of the pickle.
PICKLE_LOC = ""

def add_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def operation_gen(num):
    """Returns two lists: the first is xs, and the second is ys."""
    xs = []
    ys = []
    for _ in range(num):
        a = rand.random()*2-1
        b = rand.random()*2-1
        xs.append((a,b))
        ys.append((a+b,a*b))
    return xs, ys

def numpy_operation_gen(num):
    """The same thing as operation_gen, but returns them as numpy arrays."""
    raw_xs, raw_ys = operation_gen(num)
    return np.array(raw_xs), np.array(raw_ys)

def stringify_tensors(labels: list, tensors: list, sep='\n'):
    return sep.join([str(label) + ': ' + str(tensor) for label, tensor in zip(labels, tensors)])

def visualize_tensors(labels: list, tensors: list, start='', end='----\n'):
    return start + stringify_tensors(labels, tensors) + end

def get_all_data(pickle_location: str) -> (np.ndarray, np.ndarray):
    """Reads all the data from a single pickle. (I hope all of the data is in one pickle file)
    The number of rows read is at most num, but can be lower if it reaches EOF.
    Returns the data as two numpy arrays."""

    # TODO: I have no idea how your data is structured inside the pickle.
    # I am going to assume that in the pickle file is something like this hierarchy:

##       tuple(
##            list(
##                list containing all x-vals for first case
##                list containing all x-vals for second case
##                ...
##                list containing all x-vals for last case
##                ),
##            list(
##                list containing all y-vals for first case
##                list containing all y-vals for second case
##                ...
##                list containing all y-vals for last case
##                )
##            )
    
    # If this isn't the case you need to correct the code below as necessary. 

    raw_xs = []
    raw_ys = []
    with open(pickle_location, 'rb') as pickle_file:
        raw_xs, raw_ys = pickle.load(pickle_file)

    # OK. At this point, raw_xs and raw_ys should both be a list of lists.
    # The inner lists contain the data for each test case. 
    
    assert len(raw_xs) == len(raw_ys)

    return np.array(raw_xs), np.array(raw_ys)

def get_some_data(num, xs, ys):
    num_cases = len(xs)
    if num > num_cases:
        raise ValueError('num is {} but there are obly {} test cases'.format(num, num_cases))
    start_data = random.randint(0, num_cases-1)
    if start_data + num < num_cases:
        return xs[start_data:start_data + num], ys[start_data:start_data + num]
    else:
        return xs[start_data:] + xs[:start_data + num - num_cases], ys[start_data:] + ys[:start_data + num - num_cases]

print('###### IMPORTANT ######')
print('If you are just running this file blindly, PRESS CTRL+C NOW and read the comments in the code.')
print('You need to do things like e.g. give it the location of the files etc. where it says TODO.')
print('######  THANKS!  ######')
# once you've done this you can delete these prints and this comment

with tf.name_scope('in'):
    x = tf.placeholder(tf.float32, [None, NUM_IN], name='x') #inputs
    add_summaries(x)

with tf.name_scope('hidden1'):
    with tf.name_scope('weights'):
        W1 = tf.Variable(tf.random_uniform([NUM_IN, NUM_HIDDEN], -1.0)) #weights of hiddens
        add_summaries(W1)
    with tf.name_scope('biases'):
        b1 = tf.Variable(tf.random_uniform([NUM_HIDDEN], -1.0)) #biases
        add_summaries(b1)
    with tf.name_scope('preacts'):
        pre1 = tf.matmul(x, W1) + b1 #pre-activations of hiddens
        add_summaries(pre1)
    with tf.name_scope('acts'):
        H1 = tf.sigmoid(pre1) #activations of hiddens
        add_summaries(H1)

with tf.name_scope('out'):
    with tf.name_scope('weights'):
        W2 = tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_OUT], -1.0)) #weights of outs
        add_summaries(W2)
    with tf.name_scope('biases'):
        b2 = tf.Variable(tf.random_uniform([NUM_OUT], -1.0)) #biases of outs
        add_summaries(b2)
    with tf.name_scope('preacts'):        
        pre2 = tf.matmul(H1, W2) + b2 #pre-activations of outs
        add_summaries(pre2)
    with tf.name_scope('acts'):
        y = tf.identity(pre2) #activations of outs
        add_summaries(y)

y_ = tf.placeholder(tf.float32, [None, NUM_OUT], name='true_ys')
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
with tf.name_scope('cost'):
    mse_cost = tf.losses.mean_squared_error(y_, y)
    add_summaries(mse_cost)

train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(mse_cost)

# Get all the data from the pickle
all_xs, all_ys = get_all_data(PICKLE_LOC)

with tf.Session() as sess:
    merged_summaries = tf.summary.merge_all()
    res_writer = tf.summary.FileWriter('./results/results')
    
    tf.global_variables_initializer().run()
    for i in range(500):
        for j in range(100):
            batch_xs, batch_ys = get_some_data(50, all_xs, all_ys)
            _, cost, yr, W2r, b2r, H1r, W1r, b1r, xr = sess.run(
                (train_step, mse_cost, y, W2, b2, H1, W1, b1, x),
                feed_dict={x: batch_xs, y_: batch_ys}
                )
        if (__debug__): 
            print(visualize_tensors(
                ['y', 'W2', 'b2', 'H1', 'W1', 'b1', 'x', 'COST'],
                [yr, W2r, b2r, H1r, W1r, b1r, xr, cost]
                )) 
        
        

