import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle

X = pickle.load(open("Thay_X.pickle", 'rb'))
y = pickle.load(open("Thay_y.pickle", 'rb'))

model = Sequential()

model.add(Dense(4, input_shape=(len(X[0]),)))
model.add(Activation("relu"))

model.add(Dense(4))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(len(y[0])))
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X, y, batch_size=8, epochs=900)

model.save("Thaissa_ML")
