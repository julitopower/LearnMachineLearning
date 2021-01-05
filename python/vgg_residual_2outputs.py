#!/usr/bin/env python
import tensorflow.keras as keras
import tensorflow as tf
import basednn

def conv2d_block(units, input_layer, residual=False):
    ac = keras.layers.ELU()
    ini = 'he_normal'
    cv1 = keras.layers.Conv2D(units, (3, 3), kernel_initializer=ini, padding='same')(input_layer)
    bn1 = keras.layers.BatchNormalization()(cv1)
    ac1 = keras.layers.Activation(ac)(bn1)

    cv2 = keras.layers.Conv2D(units, (3, 3), kernel_initializer=ini, padding='same')(ac1)
    bn2 = keras.layers.BatchNormalization()(cv2)
    ac2 = keras.layers.Activation(ac)(bn2)

    cv3 = keras.layers.Conv2D(units, (3, 3), kernel_initializer=ini, padding='same')(ac2)
    bn3 = keras.layers.BatchNormalization()(cv3)
    ac3 = keras.layers.Activation(ac)(bn3)
    
    return keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(ac3)

def vgg():
    input_1 = keras.layers.Input(shape=(32, 32, 3))
    cv_block_1 = conv2d_block(64, input_1)
    cv_block_2 = conv2d_block(128, cv_block_1)
    cv_block_3 = conv2d_block(256, cv_block_2)
    avg = keras.layers.GlobalAveragePooling2D()(cv_block_3)
    actual_output = keras.layers.Dense(10)(avg)
    # This auxiliary output helps intermediate weights learn better
    # It uses the same lables as the actual output
    avg2 = keras.layers.GlobalAveragePooling2D()(cv_block_2)
    aux_output = keras.layers.Dense(10)(avg2)
    return keras.models.Model(inputs=[input_1], outputs=[actual_output, aux_output])

# Load the data
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Get the model architecture
model = vgg()
model.summary()

# Compile model: Optimizer + loss
model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss_weights=[0.9, 0.1], # We must give more importance to the actual output loss
        metrics=['accuracy'])

# Define callbacks, in this case model checkpointing and early stopping
# And LearningRateScheduler
model_checkpoint_cp = keras.callbacks.ModelCheckpoint(
    filepath='./.checkpoints/cifar10-{epoch:02d}.hd5', 
    save_best_only=True
)

lr_scheduler_cb = keras.callbacks.LearningRateScheduler(basednn.DropLearningRate(0.85, 3, 0.0015), verbose=True)

early_cb = keras.callbacks.EarlyStopping(
    patience=10,
    monitor='val_dense_accuracy',
    restore_best_weights=True
)

# vgg_inspired.py: With one output 86.2% accuracy in 18 epochs
# This file: 88% after around 60 epochs
model.fit(X_train, [y_train, y_train], epochs=1000, batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[model_checkpoint_cp, early_cb, lr_scheduler_cb])

