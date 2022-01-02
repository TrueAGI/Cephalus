"""The state kernel environment provides a sequence of inputs to a state kernel, and expects a state
tensor which captures the latent state information to be provided in return. The environment can
respond to the state tensor with a gradient indicating a desired adjustment to the state tensor,
which the state kernel will then use to train its neural models to improve future state tensor
generation."""

from abc import ABC, abstractmethod
from typing import Optional, List

import tensorflow as tf

from cephalus.support import RawTensorShape, standardize_tensor_shape, StandardizedTensorShape


class StateKernelEnvironment(ABC):
    """Abstract base class for StateKernel environments. The end user should implement this
    interface to harness a kernel in the target environment."""

    def __init__(self, input_shape: RawTensorShape):
        self._input_shape = standardize_tensor_shape(input_shape)

    @property
    def input_shape(self) -> StandardizedTensorShape:
        """The standardized tensor shape of the input space."""
        return self._input_shape

    @abstractmethod
    def get_trainable_weights(self) -> List[tf.Variable]:
        """Return a list of trainable weights which should be trained using the state prediction
        loss of the state kernel. This will typically be used for weights of environment-specific
        trainable models that preprocess inputs before they are passed to the kernel."""
        raise NotImplementedError()

    @abstractmethod
    def get_next_input(self) -> Optional[tf.Tensor]:
        """Return the next input tensor of shape self.input_shape, or None of the environment has
        terminated."""
        raise NotImplementedError()

    @abstractmethod
    def get_state_gradient(self, state: tf.Tensor) -> Optional[tf.Tensor]:
        """Return the state gradient given a state tensor."""
        raise NotImplementedError()
