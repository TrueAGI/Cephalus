from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras import Model

if TYPE_CHECKING:
    from cephalus.q.action_policies import ActionDecision


class DoubtEstimator(ABC):

    @abstractmethod
    def get_loss(self, decision: 'ActionDecision', prediction_loss: float) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_doubt(self, decision: 'ActionDecision') -> float:
        raise NotImplementedError()


class AutoencoderDoubtEstimator(DoubtEstimator):

    def __init__(self, model: Model):
        self._model = model

    def _apply_model(self, decision: 'ActionDecision'):
        target = tf.stop_gradient(decision.state)
        reconstructed = self._model(target[tf.newaxis, :])[0]
        return tf.reduce_sum(tf.square(target - reconstructed))

    def get_loss(self, decision: 'ActionDecision', prediction_loss: float):
        return self._apply_model(decision)

    def get_doubt(self, decision: 'ActionDecision') -> float:
        return self._apply_model(decision)
