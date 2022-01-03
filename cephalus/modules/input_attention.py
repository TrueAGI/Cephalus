from typing import List, Optional, TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dense, Attention

from cephalus.modules.interface import InputAttentionProvider

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel


__all__ = [
    'StandardInputAttentionProvider'
]


class StandardInputAttentionProvider(InputAttentionProvider):
    """The default input attention provider. The input attention model is trained using only the
    prediction losses provided by the other kernel modules, with no intrinsic loss. The input
    attention model is cloned from the configuration's model template, and modified to predict
    an attention key rather than a new state."""

    _query_model: Model = None
    _key_model: Model = None
    _value_model: Model = None
    _attention_layer: Attention = None

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._query_model = Sequential(
            clone_model(kernel.config.model_template).layers[1:-1],
            Dense(self.input_width)
        )
        self._key_model = Sequential(
            clone_model(kernel.config.model_template).layers[:-1],
            Dense(self.input_width)
        )
        self._value_model = clone_model(self._key_model)
        self._attention_layer = Attention()

    def get_trainable_weights(self) -> List[tf.Variable]:
        return self._key_model.trainable_weights

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.

    def attend_inputs(self, frame: 'StateFrame') -> None:
        assert frame.input_tensors
        assert frame.current_state is None
        assert frame.previous_state is not None
        assert frame.attended_input_tensor is None

        query = self._query_model(frame.previous_state[tf.newaxis, :])[tf.newaxis, :, :]

        input_tensors = tf.concat([input_tensor[tf.newaxis, :]
                                   for input_tensor in frame.input_tensors],
                                  axis=0)
        previous_state = tf.repeat(frame.previous_state[tf.newaxis, :], input_tensors.shape[0],
                                   axis=0)
        keys = self._key_model([previous_state, input_tensors])
        values = self._value_model([previous_state, input_tensors])

        # query.shape: (batch_size, query_count, channels)
        # keys.shape: (batch_size, value_count, channels)
        # values.shape: (batch_size, value_count, channels)
        assert query.shape == (1, 1, self.input_width)
        assert keys.shape == (1, len(input_tensors), self.input_width)
        assert values.shape == (1, len(input_tensors), self.input_width)

        # attended_values.shape == (batch_size, query_count, channels)
        attended_values = self._attention_layer([query, values, keys])
        assert attended_values.shape == (1, 1, self.input_width)

        attended_input = attended_values[0, 0, :]
        assert attended_input.shape == (self.input_width,)

        frame.attended_input_tensor = attended_input
