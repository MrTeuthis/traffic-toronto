import tensorflow as tf
import random as rand

# To suppress debug output, run python or python3 with the -O option:
#$ python -O tfmat.py

NUM_IN = 2
NUM_HIDDEN = 5
NUM_OUT = 2

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
    xs = []
    ys = []
    for _ in range(num):
        a = rand.random()/2.0
        b = rand.random()/2.0
        xs.append((a,b))
        ys.append((a+b,a*b))
    return xs, ys

def stringify_tensors(labels: list, tensors: list, sep='\n'):
    return sep.join([str(label) + ': ' + str(tensor) for label, tensor in zip(labels, tensors)])

def visualize_tensors(labels: list, tensors: list, start='', end='----\n'):
    return start + stringify_tensors(labels, tensors) + end

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
        pre2 = tf.sigmoid(tf.matmul(H1, W2) + b2) #pre-activations of outs
        add_summaries(pre2)
    with tf.name_scope('acts'):
        y = tf.identity(pre2) #activations of outs
        add_summaries(y)

y_ = tf.placeholder(tf.float32, [None, NUM_OUT], name='true_ys')
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
with tf.name_scope('cost'):
    mse_cost = tf.losses.mean_squared_error(y_, y)
    add_summaries(mse_cost)

train_step = tf.train.GradientDescentOptimizer(3).minimize(mse_cost)

with tf.Session() as sess:
    merged_summaries = tf.summary.merge_all()
    res_writer = tf.summary.FileWriter('./results/results')
    
    tf.global_variables_initializer().run()
    for i in range(50):
        for j in range(100):
            batch_xs, batch_ys = operation_gen(10)
            _, summaries, yr, W2r, b2r, H1r, W1r, b1r, xr = sess.run(
                (train_step, merged_summaries, y, W2, b2, H1, W1, b1, x),
                feed_dict={x: batch_xs, y_: batch_ys})
        res_writer.add_summary(summaries)
        if (__debug__): 
            print(visualize_tensors(
                ['y', 'W2', 'b2', 'H1', 'W1', 'b1', 'x'],
                [yr, W2r, b2r, H1r, W1r, b1r, xr]
                )) 

        final_cost = tf.losses.mean_squared_error(y_, y)
        if (__debug__):
            print('cost after', (i+1)*(j+1), 'iterations:',
                  sess.run(final_cost, feed_dict={x:[(0.1,0.2)] , y_: [(0.3,0.02)]}),
                  'final results:',
                  sess.run(y, feed_dict={x:[(0.1,0.2)] , y_: [(0.3,0.02)]}))
        
        

