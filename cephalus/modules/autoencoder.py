from typing import List, Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel
from cephalus.modules.interface import StateKernelModule

__all__ = [
    'StateAutoencoder'
]


class StateAutoencoder(StateKernelModule):
    """A state kernel module which adds an autoencoder loss to the kernel's state predictions."""

    _decoder = None

    def configure(self, kernel: StateKernel) -> None:
        super().configure(kernel)
        input_size = kernel.config.input_width
        self._decoder = Sequential([
            Dense(input_size + kernel.config.state_width, activation='tanh'),
            Dense(input_size + kernel.config.state_width)
        ])

    def get_trainable_weights(self) -> List[tf.Variable]:
        return self._decoder.trainable_weights

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        state_prediction = previous_frame.current_state
        flat_reconstruction = self._decoder(state_prediction[tf.newaxis, :])[0]
        flat_reconstruction_target = tf.concat([
            previous_frame.previous_state,
            previous_frame.attended_input_tensor
        ], axis=0)
        return tf.reduce_sum(tf.square(flat_reconstruction_target - flat_reconstruction))
