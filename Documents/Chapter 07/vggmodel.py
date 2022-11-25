with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(0.0, 0.01), weights_regularizer=slim.l2_regularizer(0.0005)):

# repeat twice (Conv x 2) - using "repeat"
net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='con1')
net = slim.max_pool2d(net, [2, 2], scope='pool1')

# repeat twice (Conv x 4)
net = slim.repeat(inputs, 2, slim.conv2d, 128, [3, 3], scope='con2')
net = slim.max_pool2d(net, [2, 2], scope='pool2')

# repeat three times (Conv x 7)
net = slim.repeat(inputs, 3, slim.conv2d, 256, [3, 3], scope='con3')
net = slim.max_pool2d(net, [2, 2], scope='pool3')

# repeat three times (Conv x 10)
net = slim.repeat(inputs, 3, slim.conv2d, 512, [3, 3], scope='con4')
net = slim.max_pool2d(net, [2, 2], scope='pool4')

# repeat three times (Conv x 13)
net = slim.repeat(inputs, 3, slim.conv2d, 512, [3, 3], scope='con5')
net = slim.max_pool2d(net, [2, 2], scope='pool5')

# 3 fully connected layers
net = slim.fully_connected(net, 4096, scope='fc6')
net = slim.dropout(net, 0.5, scope='dropout6')
net = slim.fully_connected(net, 4096, scope='fc7')
net = slim.dropout(net, 0.5, scope='dropout7')
net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
