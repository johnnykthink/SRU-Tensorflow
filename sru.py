import math
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class SRUCell(RNNCell):
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, _ = state

            x_size = x.get_shape().as_list()[1] # x: [1, x_size], x_size == num_units

            W_u = tf.get_variable('W_u',
                [x_size, 3 * self.num_units])

            xh = tf.matmul(x, W_u) # [1, 3 * self.num_units]

            tx, f, r = tf.split(xh, 3, 1) # [1, self.num_units] * 3
            b_f = tf.get_variable('b_f', [self.num_units])
            b_r = tf.get_variable('b_r', [self.num_units])

            f = tf.sigmoid(f + b_f)
            r = tf.sigmoid(r + b_r)

            new_c = f * c + (1 - f) * tx # element wise - multiply
            new_h = r * tf.tanh(new_c) + (1 - r) * x

            return new_h, (new_c, new_h)