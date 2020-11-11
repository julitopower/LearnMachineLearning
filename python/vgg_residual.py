import pickle as pk
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def conv2d_block(units, input_layer, residual=False):
    cv1 = keras.layers.Conv2D(units, (3, 3), padding='same')(input_layer)
    bn1 = keras.layers.BatchNormalization()(cv1)
    ac1 = keras.layers.Activation('relu')(bn1)

    cv2 = keras.layers.Conv2D(units, (3, 3), padding='same')(ac1)
    bn2 = keras.layers.BatchNormalization()(cv2)
    ac2 = keras.layers.Activation('relu')(bn2)

    cv3 = keras.layers.Conv2D(units, (3, 3), padding='same')(ac2)
    bn3 = keras.layers.BatchNormalization()(cv3)
    ac3 = keras.layers.Activation('relu')(bn3)
    
    return keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(ac3)

def vgg():
    input_1 = keras.layers.Input(shape=(32, 32, 3))
    cv_block_1 = conv2d_block(64, input_1)
    cv_block_2 = conv2d_block(128, cv_block_1)
    cv_block_3 = conv2d_block(256, cv_block_2)
    avg = keras.layers.GlobalAveragePooling2D()(cv_block_3)
    output = keras.layers.Dense(10)(avg)
    return keras.models.Model(inputs=[input_1], outputs=[output])

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

model = vgg()
model.summary()
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))

