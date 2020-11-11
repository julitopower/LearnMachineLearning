import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def vgg():
    return keras.models.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(10)
    ])   

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

model = vgg()
model.summary()
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=200,
          validation_data=(X_test, y_test))

