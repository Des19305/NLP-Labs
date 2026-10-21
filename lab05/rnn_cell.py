"""A minimal Simple RNN cell implemented with TensorFlow/Keras."""
import tensorflow as tf


class SimpleRNNCell(tf.keras.layers.Layer):
    """One recurrent step: h_t = tanh(x_t W_xh + h_(t-1) W_hh + b_h)."""

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

    def build(self, input_shape=None):
        self.W_xh = self.add_weight(
            name="W_xh",
            shape=(self.input_size, self.hidden_size),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.W_hh = self.add_weight(
            name="W_hh",
            shape=(self.hidden_size, self.hidden_size),
            initializer="orthogonal",
            trainable=True,
        )
        self.b_h = self.add_weight(
            name="b_h",
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x_t, h_prev):
        """Return hidden state for one time step."""
        return tf.math.tanh(tf.matmul(x_t, self.W_xh) +
                            tf.matmul(h_prev, self.W_hh) + self.b_h)

    def get_config(self):
        config = super().get_config()
        config.update({"input_size": self.input_size, "hidden_size": self.hidden_size})
        return config
