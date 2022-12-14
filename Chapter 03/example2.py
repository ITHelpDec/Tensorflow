import tensorflow as tf
import numpy as np

 print(tf.get_default_graph())

 g = tf.Graph()
 a = tf.constant(5)
 print(g)

 print(a.graph is g)
 print(a.graph is tf.get_default_graph())

 g1 = tf.get_default_graph()
 g2 = tf.Graph()

 print(g1 is tf.get_default_graph())

 with g2.as_default():
    print(g1 is tf.get_default_graph())

 print(g1 is tf.get_default_graph())

 a = tf.constant(5)
 b = tf.constant(2)
 c = tf.constant(3)

 d = tf.multiply(a,b)
 e = tf.add(c,b)
 f = tf.subtract(d,e)

 with tf.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs = sess.run(fetches)

 print("outs = {}".format(outs))
 print(type(outs[0]))

 c = tf.constant(4.0)
 print(c)

 c = tf.constant(4.0, dtype=tf.float64)
 print(c)
 print(c.dtype)

 x = tf.constant([1,2,3,], name='x', dtype=tf.float32)
 print(x.dtype)
 x = tf.cast(x,tf.int64)
 print(x.dtype)

# Tensor means object, tensor means scalar, vector or matrix

 c = tf.constant([[1,2,3],
                 [4,5,6]])
 2*3 matrix

 print("Python List input: {}".format(c.get_shape()))

 c = tf.constant(np.array([
                           [[1,2,3],
                           [4,5,6]],

                          [[1,1,1],
                           [2,2,2]]
                          ]))
 2*2*3 matrix

 print("3d NumPy array input: {}".format(c.get_shape()))

 sess = tf.InteractiveSession()
 c = tf.linspace(0.0, 4.0, 5)
 print("The content of 'c':\n {}\n".format(c.eval()))
 sess.close()

 A = tf.constant([[1,2,3],
                 [4,5,6]])
 print(A.get_shape())

 x = tf.constant([1,0,1])
 print(x.get_shape())

 x = tf.expand_dims(x,1)
 print(x.get_shape)

 b = tf.matmul(A,x)

 sess = tf.InteractiveSession()
 print("matmul result:\n {}".format(b.eval()))
 sess.close()

 with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    c2 = tf.constant(4,dtype=tf.int32,name='c')
 print(c1.name)
 print(c2.name)
 tried " and ', saw no difference, but that could be because of function name

 with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    with tf.name_scope("prefix_name"):
        c2 = tf.constant(4,dtype=tf.int32,name='c')
        c3 = tf.constant(4,dtype=tf.float64,name='c')

 print(c1.name)
 print(c2.name)
 print(c3.name)

 init_val = tf.random_normal((1,5),0,1)
 var = tf.Variable(init_val, name='var')
 print("pre run: \n{}".format(var))

 init = tf.global_variables_initializer()
 with tf.Session() as sess:
     sess.run(init)
     post_var = sess.run(var)

 print("\npost run: \n{}".format(post_var))

 x_data = np.random.rand(5,10)
 w_data = np.random.rand(10,1)

 with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)

    xwb = xw + b
    s = tf.reduce_max(xwb)
    with tf.Session() as sess:
        outs = sess.run(s,feed_dict={x: x_data,w: w_data})

 print("outs = {}".format(outs))

 x = tf.placeholder(tf.float32,shape=[None,3])
 y_true = tf.placeholder(tf.float32,shape=None)
 w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
 b = tf.Variable(0,dtype=tf.float32,name='bias')

 y_pred = tf.matmal(w,tf.transpose(x))+b

 loss = tf.reduce_mean(tf.square(y_true-y_pred))
 or
 loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
 loss = tf.reduce_mean(loss)

 optimizer = tf.train.GradientDescentOptimizer(learning_rate)
 train = optimizer.minimize(loss)

# ==== Create data and simulate results ====
 x_data = np.random.randn(2000,3)
 w_real = [0.3,0.5,0.1]
 b_real = -0.2

 noise = np.random.randn(1,2000)*0.1
 y_data = np.matmul(w_real,x_data.T) + b_real + noise

 NUM_STEPS = 10

 g = tf.Graph()
 wb_ = []
 with g.as_default():
     x = tf.placeholder(tf.float32,shape=[None,3])
     y_true = tf.placeholder(tf.float32,shape=None)

     with tf.name_scope('inference') as scope:
         w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
         b = tf.Variable(0,dtype=tf.float32,name='bias')
     y_pred = tf.matmul(w,tf.transpose(x)) + b

     with tf.name_scope('loss') as scope:
         loss = tf.reduce_mean(tf.square(y_true-y_pred))

     with tf.name_scope('train') as scope:
         learning_rate = 0.5
         optimizer = tf.train.GradientDescentOptimizer(learning_rate)
         train = optimizer.minimize(loss)

     Before starting, initialise the variables. We will 'run' this first
         init = tf.global_variables_initializer()
     with tf.Session() as sess:
             sess.run(init)
             for step in range(NUM_STEPS):
                 sess.run(train,{x: x_data, y_true: y_data})
                 if (step % 5 == 0):
                     print(step, sess.run([w,b]))
                     wb_.append(sess.run([w,b]))
 print(10, sess.run([w,b]))

# ==== Create data and simulate results ====
 x_data = np.random.randn(2000,3)
 w_real = [0.3,0.5,0.1]
 b_real = -0.2

 noise = np.random.randn(1,2000)*0.1
 y_data = np.matmul(w_real,x_data.T) + b_real + noise

N = 20000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Create data and simulate results =====
x_data = np.random.randn(N,3)
w_real = [0.3, 0.5,0.1]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T) + b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

NUM_STEPS = 50

g = tf.Graph()
wb_ = []
with g.as_default():
    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)

    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weights')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
    y_pred = tf.matmul(w,tf.transpose(x)) + b

    with tf.name_scope('loss') as scope:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
        loss = tf.reduce_mean(loss)

    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    # Before starting, initialise the variables. We will 'run' this first
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x: x_data, y_true: y_data})
            if (step % 5 == 0):
                print(step, sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        print(50, sess.run([w,b]))
