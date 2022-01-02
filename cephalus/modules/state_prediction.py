from typing import List, Optional, TYPE_CHECKING

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model

from cephalus.modules.interface import StatePredictionProvider

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel

__all__ = [
    'StandardStatePredictionProvider',
    'NullStatePredictionProvider',
    'UntrainedStatePredictionProvider',
]


class StandardStatePredictionProvider(StatePredictionProvider):
    """The default state prediction provider. The state model is trained using only the prediction
    losses provided by the other kernel modules, with no intrinsic loss. The state model is cloned
    directly from the configuration's model template without modification."""

    _state_model: Model = None

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._state_model = clone_model(kernel.config.model_template)

    def get_trainable_weights(self) -> List[tf.Variable]:
        return self._state_model.trainable_weights

    def get_loss(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None  # No intrinsic loss.

    def predict_state(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return self._state_model([frame.previous_state,
                                  frame.attended_input_tensor])


class NullStatePredictionProvider(StatePredictionProvider):
    """A trivial state prediction provider which ignores its gradients and simply returns the
    initial state unchanged when asked for a new state. This is useful for establishing a baseline
    in experiments, but is probably not what you want to use in production."""

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)

    def get_trainable_weights(self) -> List[tf.Variable]:
        return []

    def get_loss(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None

    def predict_state(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return self.kernel.initial_state


class UntrainedStatePredictionProvider(StatePredictionProvider):
    """A trivial state prediction provider which ignores its gradients and never trains its state
    model. This is useful for establishing a baseline in experiments, but is probably not what you
    want to use in production."""

    _state_model: Model = None

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._state_model = clone_model(kernel.config.model_template)

    def get_trainable_weights(self) -> List[tf.Variable]:
        return []

    def get_loss(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return None

    def predict_state(self, frame: 'StateFrame') -> Optional[tf.Tensor]:
        return self._state_model([frame.previous_state,
                                  frame.attended_input_tensor])
