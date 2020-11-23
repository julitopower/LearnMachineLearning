#!/usr/bin/env python
import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def vgg():

    ac = keras.layers.ELU()
    ini = 'he_normal'
    def conv(units, shape):
        return keras.layers.Conv2D(units, shape, 
                activation=ac, 
                kernel_initializer=ini,
                padding='same')
    return keras.models.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        conv(64, (3, 3)),
        conv(64, (3, 3)),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        
        conv(128, (3, 3)),
        conv(128, (3, 3)),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        
        conv(256, (3, 3)),
        conv(256, (3, 3)),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=ac, kernel_initializer=ini),
        keras.layers.Dense(256, activation=ac, kernel_initializer=ini),
        keras.layers.Dense(10)
    ])   

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
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

opt = keras.optimizers.Adam(learning_rate=0.0006)
model.compile(optimizer=opt,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

# 0.79 on test dataset after 11 epochs. I didn't let it run long enough to verify
# statiliby
model.fit(X_train, y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[early_cb, model_checkpoint_cp])

