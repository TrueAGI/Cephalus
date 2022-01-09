from abc import abstractmethod
from typing import TYPE_CHECKING, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MSE

from cephalus.modules.interface import StateKernelModule
from cephalus.names import Named

if TYPE_CHECKING:
    from cephalus.kernel import StateKernel
    from cephalus.q.action_policies import ActionDecision


class DoubtEstimator(Named):

    @abstractmethod
    def get_loss(self, decision: 'ActionDecision', prediction_loss: float) -> tf.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def get_doubt(self, decision: 'ActionDecision') -> float:
        raise NotImplementedError()


class AutoencoderDoubtEstimator(StateKernelModule, DoubtEstimator):

    def __init__(self, model: Model, *, name: str = None):
        self._model = model
        super().__init__(name=name)

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)

    def build(self) -> None:
        self._model.build(input_shape=(None, self.kernel.input_width))

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return tuple(self._model.trainable_weights)

    def get_loss(self, decision: 'ActionDecision', prediction_loss: float):
        return self._apply_model(decision)

    def get_doubt(self, decision: 'ActionDecision') -> float:
        return float(self._apply_model(decision))

    def _apply_model(self, decision: 'ActionDecision'):
        target = tf.stop_gradient(decision.state)
        reconstructed = self._model(target[tf.newaxis, :])[0]
        return MSE(target, reconstructed)
