import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
# from keras.callbacks import Tensorboard, ModelCheckpoint
# Tensorboard != TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[np.where(y_train==1)[0],:,:,:]
x_test = x_test[np.where(y_test==1)[0],:,:,:]

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_n = x_train + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_train.shape)

x_test_n = x_test + 0.5 * np.random.normal(loc=0.0, scale=0.4, size=x_test.shape)

x_train_n = np.clip(x_train_n, 0., 1.)
x_test_n = np.clip(x_test_n, 0., 1.)

inp_img = Input(shape=(32, 32, 3))

# Conv2D(number of filters / number of output images, size of each filter)
img = Conv2D(32, (3, 3), activation='relu', padding='same')(inp_img)
# 2x2 pooling layer reduces total number of pixels per channel by 4 (creates bottleneck)
img = MaxPooling2D((2, 2), padding='same')(img)

# Create another conv layer
img = Conv2D(32, (3, 3), activation='relu', padding='same')(img)

# Regain same number of units for each channel
# This is done by 4x each pixel in a pixel's near vicinity
# (repeating rows + columns
img = UpSampling2D((2, 2))(img)

# declare a conv output later to go back to three channels
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(img)

# Declare model w/ inputs and outputs
autoencoder = Model(inp_img, decoded)
# Model.load_weights(autoencoder, '/tensorflow/Chapter7/autoencoder')

# Compile model
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

tensorboard = TensorBoard(log_dir='/tensorflow/Chapter7/autoencoder', histogram_freq=0, write_graph=True, write_images=True)
model_saver = ModelCheckpoint(filepath='/tensorflow/Chapter7/autoencoder', verbose=0, period=2)

autoencoder.fit(x_train_n, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_test_n, x_test), callbacks=[tensorboard, model_saver])
