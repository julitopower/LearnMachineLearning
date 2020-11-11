import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def architecture():
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10))
    return model
            

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

model = architecture()
model.summary()
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10,
          validation_data=(X_test, y_test))

