from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any, TYPE_CHECKING, TypeVar, Generic

import tensorflow as tf

if TYPE_CHECKING:
    from cephalus.config import StateKernelConfig
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel

__all__ = [
    'StateKernelModule',
    'StatePredictionProvider',
    'RetroactiveLossProvider',
    'InputAttentionProvider',
    'Sensor',
]


Environment = TypeVar('Environment')


class StateKernelModule(Generic[Environment], ABC):
    """A pluggable module for the state kernel."""

    _kernel: Optional['StateKernel'] = None
    _loss_scale: float = 1.0

    @abstractmethod
    def configure(self, kernel: 'StateKernel') -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required."""
        assert self._kernel is None, "Kernel module is already configured."
        self._kernel = kernel

    @abstractmethod
    def get_trainable_weights(self) -> List[tf.Variable]:
        """Return a list of the trainable weights of the primary and any secondary models."""
        raise NotImplementedError()

    @abstractmethod
    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        """Return the computed loss for any models, or None if there are no trainable weights.
        Values which have already been computed (i.e. the current state) will not be recomputed. Use
        the provided gradient tape in the decision frame to get the gradients of the parameters.

        The returned loss should not be scaled. The kernel will apply loss scaling at a later
        point."""
        raise NotImplementedError()

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_new_frame_data(self, frame: 'StateFrame',
                           previous_frame: 'StateFrame' = None) -> Optional[Dict[str, Any]]:
        """Return any additional initialization information specific to the module that must be
        stored in the frame. The module's data will be stored in frame.module_data[module]."""
        return None

    @property
    def kernel(self) -> Optional['StateKernel']:
        """The state kernel this module is configured for."""
        return self._kernel

    @property
    def config(self) -> Optional['StateKernelConfig']:
        """The state kernel's configuration."""
        if self._kernel is None:
            return None
        return self._kernel.config

    @property
    def loss_scale(self) -> float:
        """A coefficient applied to the module's loss to scale its effects relative to other
        state prediction losses."""
        return self._loss_scale

    @loss_scale.setter
    def loss_scale(self, value: float) -> None:
        """A coefficient applied to the module's loss to scale its effects relative to other
        state prediction losses."""
        self._loss_scale = value

    @property
    def state_width(self) -> int:
        """The width of the state tensor for the state kernel."""
        assert self._kernel is not None
        return self._kernel.state_width

    @property
    def input_width(self) -> int:
        """The width of the input space for the state kernel."""
        assert self._kernel is not None
        return self._kernel.input_width

    @property
    def optimizer(self) -> Optional[tf.keras.optimizers.Optimizer]:
        """The optimizer used to train the state kernel's models."""
        assert self._kernel is not None
        return self._kernel.optimizer

    @property
    def initial_state(self) -> Union[tf.Tensor, tf.Variable]:
        """The initial state tensor or variable of the state kernel."""
        assert self._kernel is not None
        return self._kernel.initial_state

    @property
    def initial_state_is_trainable(self) -> bool:
        """Whether the initial ate tensor of the state kernel is a trainable variable."""
        assert self._kernel is not None
        return self._kernel.initial_state_is_trainable


class StatePredictionProvider(StateKernelModule, ABC):
    """A state kernel module which provides state predictions for its kernel. Exactly one state
    prediction provider must be configured for a given kernel. If no prediction provider has been
    added at the time the kernel is configured, a default provider will be instantiated
    automatically."""

    @abstractmethod
    def configure(self, kernel: 'StateKernel') -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required. Notify the kernel that this module will be its state prediction
        provider."""
        super().configure(kernel)
        kernel.state_prediction_provider = self

    @abstractmethod
    def predict_state(self, frame: 'StateFrame') -> tf.Tensor:
        """Predict the current state from the previous state and the current input."""
        raise NotImplementedError()


class RetroactiveLossProvider(StateKernelModule, ABC):
    """A state kernel module which provides retroactive state gradients for its kernel. At most one
    retroactive gradient provider can be configured for a given kernel."""

    def get_trainable_weights(self) -> List[tf.Variable]:
        """Return a list of the trainable weights."""
        # Models used to predict retroactive gradients should not be trained using the state loss.
        # That means we shouldn't return their trainable weights here. They'll be trained instead
        # by the call to train_retroactive_gradient()
        return []

    @abstractmethod
    def train_retroactive_loss(self, previous_frame: StateFrame,
                               current_frame: StateFrame) -> None:
        """Train the model(s) used to predict the combined state loss. (Models used for this
        purpose have a separate loss and trainable weights than those that are reported to the
        kernel, hence the separate training method.)"""
        raise NotImplementedError()


class InputAttentionProvider(StateKernelModule, ABC):
    """A state kernel module which provides input attention for its kernel. Exactly one input
    attention provider must be configured for a given kernel. If no input attention provider has
    been added at the time the kernel is configured, a default provider will be instantiated
    automatically."""

    @abstractmethod
    def configure(self, kernel: 'StateKernel') -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required. Notify the kernel that this module will be its input attention
        provider."""
        super().configure(kernel)
        kernel.input_attention_provider = self

    @abstractmethod
    def attend_inputs(self, frame: 'StateFrame') -> tf.Tensor:
        """Predict the current state from the previous state and the current input."""
        raise NotImplementedError()


class Sensor(StateKernelModule[Environment], ABC):
    """A state kernel module which provides inputs to the kernel. Inputs must have the shape
    (kernel.input_width,) to support input attention mechanisms."""

    @abstractmethod
    def get_input(self, environment: 'Environment', frame: 'StateFrame') -> Optional[tf.Tensor]:
        """Return an input tensor produced by this module."""
        raise NotImplementedError()
