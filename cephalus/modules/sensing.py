# TODO: SimpleSensor and its dependents defined here assume that the input space is dense. We need
#       to support sparse inputs with similar ease, and with appropriate mechanisms. A sparse
#       categorical input treated as if it were dense is going to bias the kernel into treating
#       categories with nearby indices as if they were more similar than categories that have
#       more widely separated indices, which is not a bias that makes sense for categorical input
#       spaces.
import warnings
from abc import ABC, abstractmethod
from functools import wraps, partial
from typing import Optional, TYPE_CHECKING, TypeVar, Union, Callable, Tuple, Dict, Hashable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from cephalus.modules.interface import Sensor
from cephalus.support import StandardizedTensorShape, size_from_shape

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
        self._mapping.build(input_shape=(None, size_from_shape(self.raw_input_shape)))
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._mapping.trainable_weights)

    def configure(self, kernel: 'StateKernel') -> None:
        self._mapping = Dense(kernel.input_width)

    def get_input(self, environment: 'Environment', frame: 'StateFrame') -> Optional[tf.Tensor]:
        raw_input = self.get_raw_input(environment, frame)
        if raw_input is None:
            return None
        raw_input_tensor = tf.cast(tf.convert_to_tensor(raw_input), self._mapping.dtype)
        flattened_input_tensor = tf.reshape(raw_input_tensor, (tf.size(raw_input_tensor),))
        return self._mapping(flattened_input_tensor[tf.newaxis, :])[0]

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.


class SensorLambda(SimpleSensor):

    def __init__(self, raw_input_shape, f: SensorFunction):
        super().__init__()
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


# A map from functions to their lambda wrappers created by the @sensor decorator, to support reuse
# of the same sensor mappings
_SENSOR_MAP: Dict[Tuple[SensorFunction, Hashable], SensorLambda] = {}


# TODO: Persistence can be implemented by providing a file path (or building one dynamically based
#       on the fully qualified module/class path of the sensor function).
# TODO: This won't work if we use the same method with different kernels. If kernel modules could
#       support multiple kernels, it would resolve that issue, but create another one: We'll need
#       a kernel registry, and the ability to mark them as single use or retire them. All of this
#       is pointing to a new design element: Persistent, automatic registries for cephalus
#       components. It would be best if this were designed up front in a systematic manner. Some
#       careful thought needs to be put into precisely when it makes sense to reuse a component, and
#       how to uniquely identify them.
def sensor(raw_input_shape: StandardizedTensorShape,
           f: SensorFunction = None, *,
           key: Hashable = None,
           single_use: bool = None) -> Union[Callable[[SensorFunction], SensorLambda],
                                             SensorLambda]:
    """Decorator for creating sensors from functions.

    Usage:
        @sensor((5, 8))
        def my_sensor(env, frame):
            sensor_reading = np.random.uniform(0, 1, (5, 8))
            return sensor_reading

        kernel.add_module(my_sensor)
    """
    if f is None:
        kwargs = {}
        if key is not None:
            kwargs.update(key=key)
        if single_use is not None:
            kwargs.update(single_use=single_use)
        return partial(sensor, raw_input_shape, **kwargs)
    if not single_use and (f, key) in _SENSOR_MAP:
        sensor_obj = _SENSOR_MAP[f, key]
        if sensor_obj.raw_input_shape != raw_input_shape:
            warnings.warn("Sensor for %s (key=%r) redefined with new shape, %s." %
                          (f, key, raw_input_shape))
        else:
            return sensor_obj
    sensor_obj = wraps(f)(SensorLambda(raw_input_shape, f))
    if not single_use:
        _SENSOR_MAP[f, key] = sensor_obj
    return sensor_obj


def retire_sensor(sensor: SensorLambda, key: Hashable = None) -> None:
    """Remove the lambda sensor created with the @sensor from the sensor cache."""
    if (sensor.f, key) in _SENSOR_MAP:
        del _SENSOR_MAP[sensor.f, key]
