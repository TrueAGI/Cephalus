from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel
from cephalus.modules.interface import StateKernelModule

__all__ = [
    'InputPrediction'
]


class InputPrediction(StateKernelModule):
    """A state kernel module which adds a prediction loss for the next input to the kernel's
    state predictions."""

    _model = None

    def configure(self, kernel: StateKernel) -> None:
        super().configure(kernel)
        self._model = Sequential([
            Dense(self.input_width + self.state_width, activation='tanh'),
            Dense(self.input_width)
        ])

    def build(self) -> None:
        self._model.build(input_shape=(None, self.state_width))
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._model.trainable_weights)

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        prediction = self._model(previous_frame.current_state[tf.newaxis, :])
        target = tf.stop_gradient(current_frame.attended_input_tensor[tf.newaxis, :])
        return tf.reduce_sum(tf.square(target - prediction))
