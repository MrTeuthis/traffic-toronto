import tensorflow as tf
import random as rand

NUM_IN = 2
NUM_HIDDEN = 10
NUM_OUT = 1

x = tf.placeholder(tf.float32, [None, NUM_IN], name='x') #inputs
W1 = tf.Variable(tf.zeros([NUM_IN, NUM_HIDDEN])) #weights of hiddens
b1 = tf.Variable(tf.zeros([NUM_HIDDEN])) #biases

H1 = tf.nn.softmax(tf.matmul(x, W1) + b1) #vals of hiddens

W2 = tf.Variable(tf.zeros([NUM_HIDDEN, NUM_OUT])) #weights of outs
b2 = tf.Variable(tf.zeros([NUM_OUT])) #biases of hiddens

y = tf.matmul(H1, W2) + b2

y_ = tf.placeholder(tf.float32, [None, NUM_OUT], name='true_ys')
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

def adder_gen(num):
    xs = []
    ys = []
    for _ in range(num):
        a = rand.random()
        b = rand.random()
        xs.append((a,b))
        ys.append((a+b,))
    return xs, ys

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(1000):
        batch_xs, batch_ys = adder_gen(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:[(1,2)] , y_: [(3,)]}))

