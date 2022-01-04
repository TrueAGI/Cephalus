from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, List, TYPE_CHECKING, TypeVar, Union, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from cephalus.modules.interface import Sensor

if TYPE_CHECKING:
    from cephalus.kernel import StateKernel
    from cephalus.frame import StateFrame


Environment = TypeVar('Environment')


class SimpleSensor(Sensor, ABC):
    _mapping: Layer = None

    @abstractmethod
    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Optional[Union[float, np.ndarray, tf.Tensor]]:
        raise NotImplementedError()

    def configure(self, kernel: 'StateKernel') -> None:
        self._mapping = Dense(kernel.input_width)

    def get_input(self, environment: 'Environment', frame: 'StateFrame') -> Optional[tf.Tensor]:
        raw_input = self.get_raw_input(environment, frame)
        if raw_input is None:
            return None
        raw_input_tensor = tf.convert_to_tensor(raw_input, dtype=self._mapping.dtype)
        flattened_input_tensor = tf.reshape(raw_input_tensor, (raw_input_tensor.size,))
        return self._mapping(flattened_input_tensor[tf.newaxis, :])[0]

    def get_trainable_weights(self) -> List[tf.Variable]:
        return self._mapping.trainable_weights

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.


class SensorLambda(SimpleSensor):

    def __init__(self, f: Callable[['Environment', 'StateFrame'],
                                   Union[None, float, np.ndarray, tf.Tensor]]):
        self.f = f

    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)

    def __call__(self, environment: 'Environment',
                 frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)


def sensor(f: Callable[['Environment', 'StateFrame'],
                       Union[float, np.ndarray, tf.Tensor]]) -> SensorLambda:
    return wraps(f)(SensorLambda(f))
