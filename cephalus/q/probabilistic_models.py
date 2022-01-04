from abc import ABC, abstractmethod
from typing import Union, List

import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.python.keras import Model
from tensorflow_probability import distributions as tfd


class ProbabilisticModelBase(ABC):

    def __init__(self, optimizer: Union[str, optimizers.Optimizer] = None):
        self.optimizer = optimizers.get(optimizer) if optimizer else None

    @property
    @abstractmethod
    def input_shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_shape(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def trainable_weights(self) -> List[tf.Variable]:
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

    def compiled_loss(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
                      samples: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return -self.log_prob(inputs, samples)

    def fit(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
            samples: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        def loss():
            return self.compiled_loss(inputs, samples)
        self.optimizer.minimize(loss, self.trainable_weights)


class ProbabilisticModel(ProbabilisticModelBase):

    def __init__(self, parameter_model: Model):
        super().__init__(parameter_model.optimizer)
        self.parameter_model = parameter_model

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

    @abstractmethod
    def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
        raise NotImplementedError()

    @property
    def trainable_weights(self) -> List[tf.Variable]:
        return self.parameter_model.trainable_weights


class DeterministicModel(ProbabilisticModel):

    def __init__(self, parameter_model: Model):
        super().__init__(parameter_model)
        self._compiled_loss = parameter_model.compiled_loss

    @property
    def output_shape(self):
        return self.parameter_model.output_shape

    def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
        return tfd.Deterministic(loc=self.parameter_model(inputs))

    def compile(self, optimizer: Union[str, optimizers.Optimizer]):
        self.optimizer = optimizers.get(optimizer)

    def compiled_loss(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
                      samples: Union[tf.Tensor, List[tf.Tensor]]) -> tf.Tensor:
        return self.parameter_model.compiled_loss or losses.MSE

    def fit(self, inputs: Union[tf.Tensor, List[tf.Tensor]],
            samples: Union[tf.Tensor, List[tf.Tensor]]) -> None:
        def loss():
            return self.compiled_loss(inputs, samples)

        self.optimizer.minimize(loss, self.parameter_model.trainable_weights)