# Running, but unexpected results

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers

DATA_DIR = '/tmp/data/'

# def model_fn(x, target, mode, params):
# model_fn must include "features" argument
# def model_fn(features, target, mode, params):
# target unexpected argument
def model_fn(features, labels, mode, params):
    # y_ = tf.cast(target, tf.float32)
    # target is not defined
    y_ = tf.cast(labels, tf.float32)
    # x_image = tf.reshape(x, [-1, 28, 28, 1])
    # x is not defined
    x_image = tf.reshape(features['x'], [-1, 28, 28, 1])

    # Conv layer 1
    conv1 = layers.convolution2d(x_image, 32, [5,5], activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool1 = layers.max_pool2d(conv1, [2,2])

    # Conv layer 2
    conv2 = layers.convolution2d(pool1, 64, [5,5], activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    pool2 = layers.max_pool2d(conv2, [2,2])

    # FC layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc1 = layers.fully_connected(pool2_flat, 1024, activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_drop = layers.dropout(fc1, keep_prob=params["dropout"], is_training=(mode == 'train'))

    # Readout layer
    y_conv = layers.fully_connected(fc1_drop, 10, activation_fn=None)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
    #    train_op = tf.contrib.layers.optimize_loss(loss=cross_entropy, global_step=tf.contrib.framework.get_global_step(), learning_rate=params["learning_rate"], optimizer="Adam")
    # tf.contrib.framework -> tf.train
    train_op = tf.contrib.layers.optimize_loss(loss=cross_entropy, global_step=tf.train.get_global_step(), learning_rate=params["learning_rate"], optimizer="Adam")

    predictions = tf.argmax(y_conv, 1)
    # return predictions, cross_entropy, train_op
    # expecting tf.estimator.EstimatorSpec
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=cross_entropy, train_op=train_op)

data = input_data.read_data_sets(DATA_DIR, one_hot=True)
x_data, y_data = data.train.images, np.int32(data.train.labels)
tf.cast(x_data, tf.float32)
tf.cast(y_data, tf.float32)

model_params = {"learning_rate": 1e-4, "dropout": 0.5}

# CNN = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
# tf.contrib.learn.Estimator is deprecated
# replace with tf.estimator.Estimators

CNN = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

print("Starting training for %s steps max" % 5000)
# CNN.fit(x=data.train.images, y=data.train.labels, batch_size=50, max_steps=5000)
# AttributeError: 'Estimator' object has no attribute 'fit'
input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':data.train.images}, y=data.train.labels, batch_size=50, num_epochs=None, shuffle=True)

CNN.train(input_fn=input_fn, steps=5000)

test_acc = 0

input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':data.test.images}, y=data.test.labels, batch_size=50, shuffle=False)

for ii in range(5):
    batch = data.test.next_batch(2000)
    predictions = list(CNN.evaluate(input_fn=input_fn))
    test_acc = test_acc + (np.argmax(batch[1],1) == predictions)#.mean() removed

print(test_acc/5)
