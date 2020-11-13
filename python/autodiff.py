#!/usr/bin/env python
"""Example usage of GradientTape for auto differentiation"""

import tensorflow as tf

# Variables are watched by default
x = tf.Variable(3.0)
# Constants are not watched by default, so tape.watch is needed
z = tf.constant(2.2, tf.float32)
# In order to invoke several types tape.gradient, we need to 
# make it persistent. Otherwise, after the first gradient call
# all the resources associated with the tape are released
with tf.GradientTape(persistent=True) as tape:
    tape.watch(z)
    y = x * x
    w = y ** z

print(tape.gradient(w, x))
print(tape.gradient(w, y))
print(tape.gradient(w, z))

del tape
