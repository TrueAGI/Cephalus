from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple

import tensorflow as tf
from tensorflow.keras import Model

from cephalus.modeled import Modeled
from cephalus.names import Named

if TYPE_CHECKING:
    from cephalus.q.action_policies import ActionDecision


class DoubtEstimator(Named):

    @abstractmethod
    def get_loss(self, decision: 'ActionDecision', prediction_loss: float) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_doubt(self, decision: 'ActionDecision') -> float:
        raise NotImplementedError()


class AutoencoderDoubtEstimator(DoubtEstimator, Modeled):

    def __init__(self, model: Model, *, name: str = None):
        self._model = model
        super().__init__(name=name)

    def build(self) -> None:
        self._model.build(input_shape=self._model.output_shape)

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._model.trainable_weights)

    def get_loss(self, decision: 'ActionDecision', prediction_loss: float):
        return self._apply_model(decision)

    def get_doubt(self, decision: 'ActionDecision') -> float:
        return self._apply_model(decision)

    def _apply_model(self, decision: 'ActionDecision'):
        target = tf.stop_gradient(decision.state)
        reconstructed = self._model(target[tf.newaxis, :])[0]
        return tf.reduce_sum(tf.square(target - reconstructed))
