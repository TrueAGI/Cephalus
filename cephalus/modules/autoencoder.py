from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from cephalus.kernel import StateKernel, StateKernelFrame, StateKernelModule
from cephalus.support import size_from_shape


class StateAutoencoder(StateKernelModule):
    """A state kernel module which adds an autoencoder loss to the kernel's state predictions."""

    _decoder = None

    def configure(self, kernel: StateKernel) -> None:
        super().configure(kernel)
        input_size = size_from_shape(kernel.config.input_shape)
        self._decoder = Sequential([
            Dense(input_size + kernel.config.state_width, activation='tanh'),
            Dense(input_size + kernel.config.state_width)
        ])

    def get_trainable_weights(self) -> List[tf.Variable]:
        return self._decoder.trainable_weights

    def get_loss(self, decision_frame: StateKernelFrame) -> Optional[tf.Tensor]:
        state_prediction = decision_frame.current_state
        flat_reconstruction = self._decoder(state_prediction[tf.newaxis, :])[0]
        flat_reconstruction_target = tf.concat([
            decision_frame.previous_state,
            tf.reshape(decision_frame.input_tensor, (decision_frame.input_tensor.size,))
        ], axis=0)
        return tf.reduce_sum(tf.square(flat_reconstruction_target - flat_reconstruction))
