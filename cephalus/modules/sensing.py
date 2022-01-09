# TODO: SimpleSensor and its dependents defined here assume that the input space is dense. We need
#       to support sparse inputs with similar ease, and with appropriate mechanisms. A sparse
#       categorical input treated as if it were dense is going to bias the kernel into treating
#       categories with nearby indices as if they were more similar than categories that have
#       more widely separated indices, which is not a bias that makes sense for categorical input
#       spaces.
import warnings
from abc import abstractmethod
from collections import deque
from functools import wraps, partial
from typing import Optional, TYPE_CHECKING, TypeVar, Union, Callable, Tuple, Dict, Iterable, \
    Deque, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer

from cephalus.modules.interface import Sensor, InputSample
from cephalus.support import StandardizedTensorShape, size_from_shape, RawTensorShape, \
    standardize_tensor_shape

if TYPE_CHECKING:
    from cephalus.kernel import StateKernel
    from cephalus.frame import StateFrame


Environment = TypeVar('Environment')


SensorFunction = Callable[['Environment', 'StateFrame'], Union[float, np.ndarray, tf.Tensor]]


class SimpleSensor(Sensor):
    _mapping: Layer = None

    def __init__(self, sensor_id: str, raw_input_shape: RawTensorShape, *,
                 loss_scale: float = None, name: str = None):
        super().__init__(loss_scale=loss_scale, name=name or sensor_id)
        self._sensor_id = sensor_id
        self._raw_input_shape = standardize_tensor_shape(raw_input_shape)
        self._sensor_embedding = None

    @abstractmethod
    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Optional[Union[float, np.ndarray, tf.Tensor]]:
        raise NotImplementedError()

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    @property
    def raw_input_shape(self) -> StandardizedTensorShape:
        return self._raw_input_shape

    @property
    def sensor_embedding(self) -> tf.Variable:
        return self._sensor_embedding

    def build(self) -> None:
        self._mapping.build(input_shape=(None, size_from_shape(self.raw_input_shape)))
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._mapping.trainable_weights + [self._sensor_embedding])

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._mapping = Dense(kernel.input_width)
        self._sensor_embedding = tf.Variable(tf.zeros(kernel.sensor_embedding_width))

    def get_inputs(self, environment: 'Environment', frame: 'StateFrame') -> Iterable[InputSample]:
        raw_input = self.get_raw_input(environment, frame)
        if raw_input is None:
            return
        raw_input_tensor = tf.cast(tf.convert_to_tensor(raw_input), self._mapping.dtype)
        flattened_input_tensor = tf.reshape(raw_input_tensor, (tf.size(raw_input_tensor),))
        value = self._mapping(flattened_input_tensor[tf.newaxis, :])[0]
        yield InputSample(self.sensor_id, self.sensor_embedding, value, frame.clock_ticks)

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.


class SensorLambda(SimpleSensor):
    """A wrapper for functions to treat them as simple sensors."""

    def __init__(self, sensor_id: str, raw_input_shape, f: SensorFunction, *,
                 loss_scale: float = None, name: str = None):
        super().__init__(sensor_id, raw_input_shape, loss_scale=loss_scale, name=name)
        self.f = f

    def get_raw_input(self, environment: 'Environment',
                      frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)

    def __call__(self, environment: 'Environment',
                 frame: 'StateFrame') -> Union[None, float, np.ndarray, tf.Tensor]:
        return self.f(environment, frame)


class SensorHistory(Sensor):
    """Wrapper for a sensor which maintains a history cache of a set length."""

    def __init__(self, wrapped: Sensor, max_length: int, *, loss_scale: float = None,
                 name: str = None):
        self._wrapped = wrapped
        self.max_length: int = max_length
        self._cache: Deque[List[InputSample]] = deque()
        super().__init__(loss_scale=loss_scale, name=name)

    @property
    def wrapped(self) -> Sensor:
        return self._wrapped

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._wrapped.configure(kernel)

    def build(self) -> None:
        self._wrapped.build()
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return super().get_trainable_weights() + self._wrapped.get_trainable_weights()

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        return self._wrapped.get_loss(previous_frame, current_frame)

    def get_inputs(self, environment: 'Environment', frame: 'StateFrame') -> Iterable[InputSample]:
        self._cache.appendleft(list(self._wrapped.get_inputs(environment, frame)))
        while len(self._cache) > self.max_length:
            self._cache.pop()
        for samples in self._cache:
            yield from samples


# A map from functions to their lambda wrappers created by the @sensor decorator, to support reuse
# of the same sensor mappings
_SENSOR_MAP: Dict[str, Union[SensorLambda, SensorHistory]] = {}


def get_default_sensor_id(f: SensorFunction) -> str:
    qualified_name = getattr(f, '__qualname__', '') or getattr(f, '__name__')
    if not qualified_name:
        raise ValueError("Please specify a sensor id for unnamed sensors, e.g. lambdas.")
    module_name = getattr(f, '__module__', '')
    return module_name + ':' + qualified_name


# TODO: Persistence can be implemented by providing a file path (or building one dynamically based
#       on the sensor id).
# TODO: This won't work if we use the same method with different kernels. If kernel modules could
#       support multiple kernels, it would resolve that issue, but create another one: We'll need
#       a kernel registry, and the ability to mark them as single use or retire them. All of this
#       is pointing to a new design element: Persistent, automatic registries for cephalus
#       components. It would be best if this were designed up front in a systematic manner. Some
#       careful thought needs to be put into precisely when it makes sense to reuse a component, and
#       how to uniquely identify them.
def sensor(raw_input_shape: StandardizedTensorShape, f: SensorFunction = None,
           sensor_id: str = None, history: int = None) \
        -> Union[Callable[[SensorFunction], SensorLambda], SensorLambda]:
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
        if sensor_id is not None:
            kwargs.update(sensor_id=sensor_id)
        return partial(sensor, raw_input_shape, **kwargs)
    if sensor_id is None:
        sensor_id = get_default_sensor_id(f)
    if sensor_id in _SENSOR_MAP:
        sensor_obj = _SENSOR_MAP[sensor_id]
        if isinstance(sensor_obj, SensorHistory):
            wrapped = sensor_obj.wrapped
        else:
            wrapped = sensor_obj
        if wrapped.f != f or wrapped.raw_input_shape != raw_input_shape:
            warnings.warn("Redefining sensor %s with function %s and shape %s.\n"
                          "Original function: %s\nOriginal shape: %s" %
                          (sensor_id, f, raw_input_shape, wrapped.f, wrapped.raw_input_shape))
        else:
            return sensor_obj
    sensor_obj = wraps(f)(SensorLambda(sensor_id, raw_input_shape, f))
    if history is not None:
        sensor_obj = SensorHistory(sensor_obj, history)
    _SENSOR_MAP[sensor_id] = sensor_obj
    return sensor_obj


def retire_sensor(sensor: Union[str, SensorFunction, SensorLambda]) -> None:
    """Remove the lambda sensor created with the @sensor from the sensor cache."""
    if isinstance(sensor, str):
        sensor_id = sensor
    elif isinstance(sensor, SensorLambda):
        sensor_id = sensor.sensor_id
    else:
        sensor_id = get_default_sensor_id(sensor)
    if sensor_id in _SENSOR_MAP:
        del _SENSOR_MAP[sensor_id]
