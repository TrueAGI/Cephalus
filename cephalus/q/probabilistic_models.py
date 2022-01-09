from abc import abstractmethod
from typing import Union, List, Tuple, Type

import tensorflow as tf
from tensorflow.keras import optimizers, losses, Model
from tensorflow.keras.layers import Layer
from tensorflow_probability import distributions as tfd

from cephalus.modeled import Modeled


class ProbabilisticModelBase(Modeled):

    def __init__(self, optimizer: Union[str, optimizers.Optimizer] = None, *, name: str = None):
        self.optimizer = optimizers.get(optimizer) if optimizer else None
        super().__init__(name=name)

    @property
    @abstractmethod
    def input_shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
        raise NotImplementedError()

    def mean(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return self(inputs).mean

    def sample(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> Union[tf.Tensor,
                                                                         List[tf.Tensor]]:
        return self(inputs).sample()

    def prob(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
             sample: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return self(inputs).prob(sample)

    def log_prob(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
                 sample: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return self(inputs).log_prob(sample)

    def compile(self, optimizer: Union[str, optimizers.Optimizer]):
        self.optimizer = optimizers.get(optimizer)

    @staticmethod
    def distribution_loss(distribution: tfd.Distribution,
                          samples: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return -distribution.log_prob(samples)

    def compiled_loss(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
                      samples: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return self.distribution_loss(self(inputs), samples)

    def fit(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
            samples: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        def loss():
            return self.compiled_loss(inputs, samples)
        self.optimizer.minimize(loss, self.get_trainable_weights())


class ProbabilisticModel(ProbabilisticModelBase):

    def __init__(self, parameter_model: Model, distribution_type: Type[tfd.Distribution], *,
                 name: str = None):
        self.parameter_model = parameter_model
        self.distribution_type = distribution_type
        super().__init__(getattr(parameter_model, 'optimizer', None), name=name)

    @property
    def input_shape(self):
        return self.parameter_model.input_shape

    @property
    def output_shape(self):
        output_shape = self.parameter_model.output_shape
        if isinstance(output_shape, list):
            # We assume that each output corresponds to a parameter of the distribution, and that
            # they all have the same shape. If your use case doesn't fit these assumptions, you'll
            # need to override the output_shape property.
            output_shape = output_shape[0]
        return output_shape

    def build(self) -> None:
        assert self.parameter_model.built
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self.parameter_model.trainable_weights)

    def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
        parameters = self.parameter_model(inputs)
        if isinstance(parameters, tf.Tensor):
            return self.distribution_type(parameters)
        else:
            return self.distribution_type(*parameters)


class DeterministicModel(ProbabilisticModel):

    def __init__(self, parameter_model: Model, *, name: str = None):
        if not isinstance(parameter_model, Layer):
            raise TypeError(type(parameter_model), Layer)
        self._compiled_loss = getattr(parameter_model, 'compiled_loss', None)
        super().__init__(parameter_model, tfd.Deterministic, name=name)

    @property
    def output_shape(self):
        return self.parameter_model.output_shape

    def compile(self, optimizer: Union[str, optimizers.Optimizer]):
        self.optimizer = optimizers.get(optimizer)

    def distribution_loss(self, distribution: tfd.Distribution,
                          samples: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        loss_func = self.parameter_model.compiled_loss or losses.MSE
        return loss_func(distribution.mean()[tf.newaxis], samples[tf.newaxis])

    def fit(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
            samples: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        def loss():
            return self.compiled_loss(inputs, samples)

        self.optimizer.minimize(loss, self.parameter_model.trainable_weights)
