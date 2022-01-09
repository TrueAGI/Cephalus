from typing import Optional, TYPE_CHECKING, Tuple, Union

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Attention
from tensorflow.keras.models import clone_model

from cephalus.modules.interface import InputAttentionProvider, InputSample

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
    _default_input: tf.Variable = None
    _default_sensor_embedding: tf.Variable = None

    @property
    def sensor_embedding_width(self) -> int:
        return self.kernel.sensor_embedding_width

    @property
    def kv_input_width(self) -> int:
        return self.state_width + self.sensor_embedding_width + self.input_width + 1

    @property
    def default_input(self) -> Union[tf.Tensor, tf.Variable]:
        """The default input tensor or variable used by the state kernel when there are no
        inputs."""
        assert self._default_input is not None
        return self._default_input

    @property
    def default_sensor_embedding(self) -> Union[tf.Tensor, tf.Variable]:
        """The sensor embedding for the default input used when there are no inputs."""
        assert self._default_sensor_embedding is not None
        return self._default_sensor_embedding

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._query_model = Sequential(
            clone_model(kernel.config.model_template).layers[:-1] +
            [Dense(self.input_width)]
        )
        self._key_model = Sequential(
            clone_model(kernel.config.model_template).layers[:-1] +
            [Dense(self.input_width)]
        )
        self._value_model = clone_model(self._key_model)
        self._attention_layer = Attention()
        self._default_input = tf.Variable(tf.zeros(kernel.input_width, dtype=kernel.dtype))
        self._default_sensor_embedding = tf.Variable(tf.zeros(kernel.sensor_embedding_width,
                                                              dtype=kernel.dtype))

    def build(self) -> None:
        self._query_model.build(input_shape=(None, self.state_width))
        self._key_model.build(input_shape=(None, self.kv_input_width))
        self._value_model.build(input_shape=(None, self.kv_input_width))
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(
            self._query_model.trainable_weights +
            self._key_model.trainable_weights +
            self._value_model.trainable_weights +
            [self._default_input, self._default_sensor_embedding]
        )

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.

    def get_annotated_tensor(self, sample: 'InputSample', frame: 'StateFrame'):
        age = frame.clock_ticks - sample.time_stamp
        return tf.concat([
            tf.convert_to_tensor([1.0 - 1.0 / (1.0 + age)], dtype=self.kernel.dtype),
            sample.sensor_embedding,
            sample.value
        ], axis=-1)

    def attend_inputs(self, frame: 'StateFrame') -> None:
        assert frame.input_samples is not None
        assert frame.current_state is None
        assert frame.previous_state is not None
        assert frame.attended_input_tensor is None

        input_tensors = []
        for sample in frame.input_samples:
            input_tensors.append(self.get_annotated_tensor(sample, frame))

        if not input_tensors:
            sample = InputSample('<DEFAULT>', self.default_sensor_embedding, self.default_input, 0)
            input_tensors.append(self.get_annotated_tensor(sample, frame))

        query = self._query_model(frame.previous_state[tf.newaxis, :])[tf.newaxis, :, :]

        input_tensors = tf.concat([input_tensor[tf.newaxis, :] for input_tensor in input_tensors],
                                  axis=0)
        previous_state = tf.repeat(frame.previous_state[tf.newaxis, :], input_tensors.shape[0],
                                   axis=0)
        kv_in = tf.concat([previous_state, input_tensors], axis=-1)
        keys = self._key_model(kv_in)[tf.newaxis, :, :]
        values = self._value_model(kv_in)[tf.newaxis, :, :]

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
