#!/usr/bin/env python
import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


class ConvBlock(keras.layers.Layer):
    """A Layer using Conv2d, BatchNorm, and RElu"""
    def __init__(self, units):
        """Builds the layer

        :param units: Int. Number of filters in the Conv2D layers
        """
        def conv_block():
            return [
                keras.layers.Conv2D(units, (3, 3), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu')
            ]
        super().__init__()
        self.units = []
        self.units.extend(conv_block())
        self.units.extend(conv_block())
        self.units.extend(conv_block())
        self.units.append(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))

    def call(self, inputs):
        """Forward pass on the layer"""
        Z = inputs
        for layer in self.units:
            Z = layer(Z)
        return Z


class VGG(keras.Model):
    def __init__(self):
        super().__init__()
        self.units = []
        self.units.append(ConvBlock(64))
        self.units.append(ConvBlock(128))
        self.units.append(ConvBlock(256))
        self.units.append(keras.layers.GlobalAveragePooling2D())
        self.units.append(keras.layers.Dense(10))

    def call(self, inputs):
        Z = inputs
        for layer in self.units:
            Z = layer(Z)
        return Z


# Load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)
batch_size = 32
# Get the model architecture
model = VGG()

# Compile model: Optimizer + loss
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

# Define callbacks, in this case model checkpointing and early stopping
model_checkpoint_cp = keras.callbacks.ModelCheckpoint(
    filepath='./.checkpoints/cifar10-{epoch:02d}.hd5', 
    save_best_only=True
)

early_cb = keras.callbacks.EarlyStopping(
    patience=3,
    monitor='val_accuracy',
    restore_best_weights=True
)
model.fit(X_train, y_train, epochs=20, batch_size=batch_size,
          validation_data=(X_test, y_test),
          callbacks=[model_checkpoint_cp, early_cb])

