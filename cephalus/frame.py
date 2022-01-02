from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any, TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from cephalus.modules.interface import StateKernelModule

__all__ = [
    'StateFrame'
]


@dataclass
class StateFrame:
    """A collection of all the information collected about a state prediction which may be required
    to train the kernel. Kernels may subclass this and add their own fields."""

    previous_state: Union[tf.Tensor, tf.Variable]
    tape: Optional[tf.GradientTape]

    input_tensors: List[tf.Tensor] = None
    attended_input_tensor: tf.Tensor = None
    current_state: Optional[tf.Tensor] = None
    current_state_gradient: Optional[tf.Tensor] = None
    future_gradient_prediction: Optional[tf.Tensor] = None
    trained: bool = False

    module_data: Dict['StateKernelModule', Dict[str, Any]] = field(default_factory=dict)
