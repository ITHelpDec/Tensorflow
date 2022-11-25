import tensorflow as tf
from tensorflow.contrib import learn

from sklearn import datasets, metrics, preprocessing
boston = datasets.load_boston()
x_data = preprocessing.StandardScaler().fit_transform(boston.data)
y_data = boston.target

x = tf.placeholder(tf.float64, shape=(None,13))
y_true = tf.placeholder(tf.float64, shape=(None))

with tf.name_scope('inference') as scope:
    w = tf.Variable(tf.zeros([1,13], dtype=tf.float64, name='weights'))
    b = tf.Variable(0, dtype=tf.float64, name='bias')
    y_pred = tf.matmul(w, tf.transpose(x)) + b
    # y = wx + b
    # just have to transpose x in order to multiply matrices

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    # the is the mean squared error

# with tf.name_scope('train') as scope:
#    learning_rate = 0.1
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#    train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(train,{x:x_data, y_true:y_data})
    # MSE = sess.run(loss,{x:x_data, y_true:y_data})
    NUM_STEPS = 200
    MINIBATCH_SIZE = 506

    feature_columns = learn.infer_real_valued_cloumns_from_input(xdata)

    reg = learn.LinearRegressor(feature_columns=feature_columns, optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1))

    reg.fit(x_data, boston.target, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)
    MSE = reg.evaluate(x_data, boston.target, steps=1)
print(MSE)
