import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Define some parameters
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128

# Where to save TensorBoard model summaries
LOG_DIR = "/tensorflow/logs/RNN_with_summaries"

# Create placeholders for inputs, labels
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

batch_x, batch_y = mnist.train.next_batch(batch_size)
# Reshape data to get 28 sequences of 28 pixels
batch_x = batch_x.reshape((batch_size, time_steps, element_size))

# This helper function, taken from the official TensorFlow documentation
# simply add some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# Weights and bias for input and hidden layer
with tf.name_scope('run_weights'):
    with tf.name_scope("W_x"):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope("W_h"):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope("Bias"):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)

# Implement vanilla RNN step (formula)
# O'Reilly did it duble backwards??
# 1) order of matrix addition
# 2) order of variables in tf.matmul
# could this be the reason why the programme didn't run?

# def rnn_step(previous_hidden_state,x):
#    current_hidden_state = tf.tanh(tf.matmul(x, Wx) + tf.matmul(previous_hidden_state, Wh) + b_rnn)
#    return current_hidden_state

def rnn_step(previous_hidden_state,x):
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
    return current_hidden_state

# Processing inputs to work with scan fucntion
# Current input shape: (batch_size, time_steps, element_size)
# 0 = batch_size, 1 = time_steps, 2 = element size
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# Current input shape now: (time_steps, batch_size, element_size)
# 1 = time_steps, 0 = batch_size, 2=element_size


initial_hidden = tf.zeros([batch_size,hidden_layer_size])
# Getting all state vectors across time
all_hidden_states = tf.scan(rnn_step, processed_input, initializer=initial_hidden, name='states')

# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope("W_linear"):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))
        variable_summaries(Wl)
    with tf.name_scope("Bias_linear"):
        bl = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
        variable_summaries(bl)

# Apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)
    # Get last output
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # Using RMSPropOptimizer
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(output,1))
    
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries
merged = tf.summary.merge_all()

# Get a small test set
test_data = mnist.test.images[:batch_size].reshape((-1, time_steps,element_size))
test_label = mnist.test.labels[:batch_size]

with tf.Session() as sess:
    # Write summaries to LOG_DIR -- used by TensorBoard
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train', graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test', graph=tf.get_default_graph())
    
    sess.run(tf.global_variables_initializer())

    for i in range(10000):

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 sequences of 28 pixels
        # was missing batch_x from tutorial
        batch_x = batch_x.reshape((batch_size, time_steps, element_size))
        summary,_ = sess.run([merged,train_step],feed_dict={_inputs:batch_x, y:batch_y})
        # Add to summaries
        train_writer.add_summary(summary, i)

        if i % 1000 == 0:
            acc,loss, = sess.run([accuracy,cross_entropy],feed_dict={_inputs: batch_x, y: batch_y})
            print("Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        if i % 10:
            # Calculate accuracy for 128 MNIST test images and
            # add to summaries
            summary, acc = sess.run([merged, accuracy],feed_dict={_inputs:test_data, y: test_label})
            test_writer.add_summary(summary, i)

    test_acc = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
    print("Test Accuracy:", test_acc)

