import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
# a "Dense" layer is a fully connected layer

model = Sequential()

model.add(Dense(units=64, input_dim=784))
# units = number of output units
# input_dim = shape of input i.e. 784x64
model.add(Activation('softmax'))

# mode = Sequential([Dense(64, input_shape=(784,), activation='softmax')])

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.8, nesterov=True)
# lr = learning rate

from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

# model.fit(x_train, y_train, epochs=10, batch_size=64, callbacks=[TensorBoard(log_dir='/models/autoencoder',)early_stop])
# syntax error from tutorial
# comma should be outside the bracket following directory declaration

model.fit(x_train, y_train, epochs=10, batch_size=64, callbacks=[TensorBoard(log_dir='/models/autoencoder'), early_stop])

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
classes = model.predict(x_test, batch_size=64)

# Functional Model

# Create input Tensor
inputs = Input(shape=(784,))

# Define our model
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop', loss='catergorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=64)
classes = mode.predict(x_test, batch_size=64)

