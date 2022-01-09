from typing import Optional, TYPE_CHECKING, Tuple, Union, Callable

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
    _input_attention_function: Callable = None

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

        @tf.function
        def input_attention_function(current_state, input_tensors):
            query = self._query_model(current_state[tf.newaxis, :])[tf.newaxis, :, :]

            current_state_repeated = tf.repeat(current_state[tf.newaxis, :],
                                               tf.shape(input_tensors)[0],
                                               axis=0)
            kv_in = tf.concat([current_state_repeated, input_tensors], axis=-1)
            keys = self._key_model(kv_in)[tf.newaxis, :, :]
            values = self._value_model(kv_in)[tf.newaxis, :, :]

            # query.shape: (batch_size, query_count, channels)
            # keys.shape: (batch_size, value_count, channels)
            # values.shape: (batch_size, value_count, channels)
            tf.assert_equal(tf.shape(query), [1, 1, self.input_width])
            tf.assert_equal(tf.shape(keys), [1, len(input_tensors), self.input_width])
            tf.assert_equal(tf.shape(values), [1, len(input_tensors), self.input_width])

            # attended_values.shape == (batch_size, query_count, channels)
            attended_values = self._attention_layer([query, values, keys])
            tf.assert_equal(tf.shape(attended_values), [1, 1, self.input_width])

            attended_input = attended_values[0, 0, :]
            tf.assert_equal(tf.shape(attended_input), [self.input_width])

            return attended_input

        self._input_attention_function = input_attention_function

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

        input_tensors = tf.concat([input_tensor[tf.newaxis, :] for input_tensor in input_tensors],
                                  axis=0)

        attended_input = self._input_attention_function(frame.previous_state, input_tensors)

        frame.attended_input_tensor = attended_input
