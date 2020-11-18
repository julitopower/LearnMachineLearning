#!/usr/bin/env python
import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from basednn import DNNLayer, DNNModel, EarlyStopper

class ConvLiteBlock(DNNLayer):
    def __init__(self, units, shapes=(3, 3), strides=(1, 1), activation='relu', padding='same'):
        super().__init__()
        self.c1 = keras.layers.Conv2D(units, kernel_size=shapes, strides=strides, padding=padding)
        self.bn = keras.layers.BatchNormalization(axis=-1)
        self.ac = keras.layers.Activation(activation=activation)

    def call(self, inputs):
        return self.ac(self.bn(self.c1(inputs)))

class InceptionLiteBlock(DNNLayer):
    def __init__(self, units1, units2):
        super().__init__()
        self.c1 = ConvLiteBlock(units1, shapes=(1, 1), strides=(1, 1), padding='same')
        self.c2 = ConvLiteBlock(units1, shapes=(3, 3), strides=(1, 1), padding='same')

    def call(self, inputs):
        Z = inputs
        return tf.concat([self.c1(Z), self.c2(Z)], axis=-1)


class DownsampleLiteBlock(DNNLayer):
    def __init__(self, units):
        super().__init__()
        self.c1 = ConvLiteBlock(units, shapes=(3, 3), strides=(2, 2), padding='valid')
        self.mp = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))

    def call(self, inputs):
        Z = inputs
        return tf.concat([self.c1(Z), self.mp(Z)], axis=3)

class GoogleNetLite(DNNModel):
    def __init__(self):
        super().__init__()
        (
        self +
        ConvLiteBlock(units=96, shapes=(3, 3), strides=(1, 1), activation='relu') +
        InceptionLiteBlock(32, 32) +
        InceptionLiteBlock(32, 48) +
        DownsampleLiteBlock(80) +
        InceptionLiteBlock(112, 48) +
        InceptionLiteBlock(96, 64) +
        InceptionLiteBlock(80, 80) +
        InceptionLiteBlock(48, 96) +
        DownsampleLiteBlock(96) +
        InceptionLiteBlock(176, 160) +
        InceptionLiteBlock(176, 160) +
        keras.layers.AveragePooling2D(pool_size=(7, 7)) +
        keras.layers.Dropout(rate=0.5) +
        keras.layers.Flatten() +
        keras.layers.Dense(10)
        )

def Inceptionblock(DNNLayer):
    def __init__(self, units):
        super().__init__()
        c1 = keras.layers.Conv2D(units[0], shape=(1, 1), padding='same', activation='relu')
        c2 = keras.layers.Conv2D(units[1], shape=(1, 1), padding='same', activation='relu')
        mp = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')#, activation='relu')
        c3 = keras.layers.Conv2D(units[2], shape=(1, 1), padding='same', activation='relu')
        c4 = keras.layers.Conv2D(units[3], shape=(3, 3), padding='same', activation='relu')
        c5 = keras.layers.Conv2D(units[4], shape=(5, 5), padding='same', activation='relu')
        c6 = keras.layers.Conv2D(units[5], shape=(1, 1), padding='same', activation='relu')

    def call(self, inputs):
        Z = inputs
        path1 = c1(Z)
        path2 = c4(c2(Z))
        path3 = c5(c3(Z))
        path4 = c6(mp(Z))
        return tf.concat([path1, path2, path3, path4], axis=3)
        #return tf.concat([path1, path2, path3], axis=3)


class GoogleNet(DNNModel):
    """A small variation of GoogleNet. It only achieves 70% accuracy
    
    It is too complex for the CIFAR10 images. I had to remove some of the
    stride=2 on the pooling layer so that the model would actually work.
    """
    def __init__(self):
        super().__init__()
        (
        self + keras.layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu') +
        keras.layers.MaxPooling2D((3, 3), strides=2, padding='same') +
        tf.nn.local_response_normalization +
        keras.layers.Conv2D(64, (1, 1), strides=1, padding='same', activation='relu') +
        keras.layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu') +
        tf.nn.local_response_normalization +
        keras.layers.MaxPooling2D((3, 3), strides=1, padding='same') +
        Inceptionblock([64, 96, 16, 128, 32, 32]) +
        Inceptionblock([128, 128, 32, 192, 96, 64]) +
        keras.layers.MaxPooling2D((3, 3), strides=1) +
        Inceptionblock([192, 96, 16, 208, 48, 64]) +
        Inceptionblock([160, 112, 23, 224, 64, 64]) +
        Inceptionblock([128, 128, 24, 256, 64, 64]) +
        Inceptionblock([112, 144, 32, 288, 64, 64]) +
        Inceptionblock([256, 160, 32, 320, 128, 128]) +
        keras.layers.MaxPooling2D((3, 3), strides=1) +
        Inceptionblock([256, 160, 32, 320, 128, 128]) +
        Inceptionblock([384, 192, 48, 384, 128, 128]) +
        keras.layers.GlobalAveragePooling2D() +
        keras.layers.Dropout(rate=0.4) +
        keras.layers.Dense(10)
        )


# Load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
print(X_train)
X_train = tf.cast(X_train, tf.float32) / 255.0
X_test = tf.cast(X_test, tf.float32) / 255.0
batch_size = 16
# Get the model architecture
model = GoogleNetLite()
optimizer = keras.optimizers.Adam(learning_rate=0.0005)
# Compile model: Optimizer + loss
model.compile(optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

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
model.fit(X_train, y_train, epochs=1000, batch_size=batch_size,
          validation_data=(X_test, y_test),
          callbacks=[model_checkpoint_cp, early_cb])

