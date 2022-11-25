import tensorflow as tf

x = tf.constant(4.0)

with tf.GradientTape() as t:
    t.watch(x)
    y = x * x
print()
print()
print(t.gradient(y, x))