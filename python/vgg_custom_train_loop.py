#!/usr/bin/env python
import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

################################################################################
#
# Functions to define model architecture
#
################################################################################
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
        self.units.append(keras.layers.MaxPooling2D((2, 2), 
                          strides=(2, 2), 
                          padding='same'))

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


################################################################################
# 
# End of functions to define model architecture
#
################################################################################

# Load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = tf.cast(X_train, tf.float32)
X_test = tf.cast(X_test, tf.float32)

batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.batch(batch_size=batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size=batch_size)

# Get the model architecture
model = VGG()

epochs = 20 

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
mean_loss = tf.keras.metrics.Mean()
train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
val_loss = tf.keras.metrics.Mean()
val_acc = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_fn(X, y):
    """Train one step of an epoch.

    Marked as tf.function so that graph compilation takes place
    and execution is faster
    """

    # Forward pass + loss calculation, with autodiff enabled
    with tf.GradientTape() as tape:
        logits = model(X)
        loss = loss_fn(y, logits)
    step_mean_loss = tf.reduce_mean(loss)

    # Backprop and metrics collection
    mean_loss(step_mean_loss)
    grad = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grad, model.trainable_weights))
    train_acc.update_state(y, logits)

@tf.function
def test_fn(X_val, y_val):
    logits = model(X_val)
    loss = tf.reduce_mean(loss_fn(y_val, logits))
    val_loss.update_state(loss)
    val_acc.update_state(y_val, logits)

################################################################################
# 
# Actual training loop start here
#
################################################################################
for epoch in range(epochs):
    for step, (X, y) in enumerate(train_dataset.as_numpy_iterator()):
        train_fn(X, y)

        # Step Reporting
        if step % 10 == 0:
            pass
            print(f'\rStep {step}: train_loss {mean_loss.result():.4f} - train_acc {train_acc.result():.4f}', end='', flush=True)
    # Caculate validation loss and accuracy


    for (X_val, y_val) in test_dataset.as_numpy_iterator():
        test_fn(X_val, y_val)

    # Epoch reporting
    print(f'\nEpoch {epoch}: train_loss {mean_loss.result():.4f} - train_acc {train_acc.result():.4f} - ', end='', flush=True)
    print(f'val_loss {val_loss.result():.4f} - val_acc {val_acc.result():.4f} - ', flush=True)
    # Metrics cleanup in preparation for the next epoch
    mean_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()

