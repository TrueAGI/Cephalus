from dataclasses import dataclass

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

__all__ = [
    'StateKernelConfig'
]


@dataclass
class StateKernelConfig:
    """The configuration data for a state kernel. Includes basic information necessary for the
    kernel and its modules to construct, train, and utilize their neural models."""

    # The width of the state space. The state space is always 1-dimensional. Tasks which require
    # other shapes must transform their inputs from this shape.
    state_width: int

    # The with of the input space. The input space is always 1-dimensional. Sensors which have
    # other shapes must transform their outputs to this shape.
    input_width: int

    # A model template with a structure suitable for predicting the next state. The model must
    # accept two inputs, one for previous state and one for current (combined) inputs, in that
    # order. It must return a single output, for the current state. The combined input tensor is
    # computed using attention over all available inputs, so its shape will be the same as the
    # individual input tensors.
    model_template: Model

    # The optimizer to be used for all the models.
    optimizer: Optimizer
