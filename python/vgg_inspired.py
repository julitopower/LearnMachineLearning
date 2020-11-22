#!/usr/bin/env python
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


# Define callbacks, in this case model checkpointing and early stopping
model_checkpoint_cp = keras.callbacks.ModelCheckpoint(
    filepath='./.checkpoints/cifar10-{epoch:02d}.hd5', 
    save_best_only=True
)

early_cb = keras.callbacks.EarlyStopping(
    patience=100,
    monitor='val_accuracy',
    restore_best_weights=True
)

model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

# 74% accuracy on test dataset after 7 epochs. After 70+ epochs weights get corrupt
# and accuracy remains at 10% from that epoch onwards
model.fit(X_train, y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[early_cb, model_checkpoint_cp])

