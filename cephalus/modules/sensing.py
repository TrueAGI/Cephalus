# TODO: SimpleSensor and its dependents defined here assume that the input space is dense. We need
#       to support sparse inputs with similar ease, and with appropriate mechanisms. A sparse
#       categorical input treated as if it were dense is going to bias the kernel into treating
#       categories with nearby indices as if they were more similar than categories that have
#       more widely separated indices, which is not a bias that makes sense for categorical input
#       spaces.


from abc import ABC, abstractmethod
from functools import wraps, partial
from typing import Optional, TYPE_CHECKING, TypeVar, Union, Callable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from cephalus.modules.interface import Sensor
from cephalus.support import StandardizedTensorShape

if TYPE_CHECKING:
    from cephalus.kernel import StateKernel
    from cephalus.frame import StateFrame


Environment = TypeVar('Environment')


SensorFunction = Callable[['Environment', 'StateFrame'], Union[float, np.ndarray, tf.Tensor]]


class SimpleSensor(Sensor, ABC):
    _mapping: Layer = None

    @property
    @abstractmethod
    def raw_input_shape(self) -> StandardizedTensorShape:
        raise NotImplementedError()

    @abstractmethod
    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Optional[Union[float, np.ndarray, tf.Tensor]]:
        raise NotImplementedError()

    def build(self) -> None:
        self._mapping.build(input_shape=(None,) + self.raw_input_shape)
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._mapping.trainable_weights)

    def configure(self, kernel: 'StateKernel') -> None:
        self._mapping = Dense(kernel.input_width)

    def get_input(self, environment: 'Environment', frame: 'StateFrame') -> Optional[tf.Tensor]:
        raw_input = self.get_raw_input(environment, frame)
        if raw_input is None:
            return None
        raw_input_tensor = tf.convert_to_tensor(raw_input, dtype=self._mapping.dtype)
        flattened_input_tensor = tf.reshape(raw_input_tensor, (tf.size(raw_input_tensor),))
        return self._mapping(flattened_input_tensor[tf.newaxis, :])[0]

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.


class SensorLambda(SimpleSensor):

    def __init__(self, raw_input_shape, f: SensorFunction):
        self._raw_input_shape = raw_input_shape
        self.f = f

    @property
    def raw_input_shape(self) -> StandardizedTensorShape:
        return self._raw_input_shape

    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)

    def __call__(self, environment: 'Environment',
                 frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)


def sensor(raw_input_shape: StandardizedTensorShape,
           f: SensorFunction = None) -> Union[Callable[[SensorFunction], SensorLambda],
                                              SensorLambda]:
    if f is None:
        return partial(sensor, raw_input_shape)
    return wraps(f)(SensorLambda(raw_input_shape, f))
