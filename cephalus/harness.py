"""A harness which links a state kernel to its environment, coordinating their interactions."""

from dataclasses import dataclass
from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from cephalus.environment import StateKernelEnvironment
from cephalus.kernel import StateKernel, StateKernelConfig, StateKernelFrame


@dataclass
class StateKernelHarnessConfig:
    """The configuration for a state kernel harness."""

    # The width of the state space. The state space is always 1-dimensional.
    state_width: int

    # A model template with a structure suitable for predicting the next state for the environment.
    model_template: Model

    # The optimizer to be used for all the models.
    optimizer: Optimizer

    # The coefficient to multiply predicted gradients for future states that have been propagated
    # back to the current state. Acts as a future discount rate.
    future_gradient_coefficient: float

    # Whether to divide the sum of the current true gradient and the discounted future gradient by
    # (1 + future_gradient_coefficient). If true, effectively normalizes their sum to the natural
    # range of the true gradient.
    stabilized_gradient: bool


class StateKernelHarness:
    """Harnesses a state kernel to make state predictions within an environment."""

    def __init__(self, config: StateKernelHarnessConfig, kernel: StateKernel,
                 environment: StateKernelEnvironment):
        self._config = config
        self._kernel = kernel
        self._environment = environment
        self._previous_frame: Optional[StateKernelFrame] = None

        kernel_config = StateKernelConfig(
            state_width=config.state_width,
            input_shape=environment.input_shape,
            model_template=config.model_template,
            optimizer=config.optimizer,
            additional_trainable_weights=tuple(environment.get_trainable_weights()),
        )
        kernel.configure(kernel_config)

    def run(self, steps: int = None, terminate: Callable[[], bool] = None) -> bool:
        """Run the kernel in the environment. If steps is provided, run for at most that many
        steps. If terminate callback is provided, call it just before each step and stop if it
        returns True. Return whether the environment still expects more steps."""
        step = 0
        while True:
            if steps is not None and step >= steps:
                break
            if terminate is not None and terminate():
                break
            step += 1
            if not self.step():
                return False
        return True

    def step(self) -> bool:
        """Run the kernel for a single step in the environment. Return whether the environment
        still expects more steps."""
        frame = self._kernel.new_decision_frame(self._previous_frame)
        input_tensor = self._environment.get_next_input()
        if input_tensor is not None:
            self._kernel.accept_input(frame, input_tensor)

        self._combine_gradients(frame)

        # We train even if there were no external gradients, as some kernels have internally-induced
        # gradients.
        self._kernel.train(self._previous_frame)

        # Ask the environment for the current state's gradient and record it for later training.
        state_gradient = self._environment.get_state_gradient(frame.current_state)
        if state_gradient is not None:
            assert not tf.reduce_any(tf.math.is_nan(state_gradient))
            frame.current_state_gradient = state_gradient

        self._previous_frame = frame

        # Return whether the environment still expects more steps.
        return input_tensor is not None

    def _combine_gradients(self, new_frame: StateKernelFrame) -> None:
        # Gather gradients for the previous state.
        state_gradients = []
        state_gradient_weights = []
        if (self._previous_frame is not None and
                self._previous_frame.current_state_gradient is not None):
            state_gradients.append(self._previous_frame.current_state_gradient)
            state_gradient_weights.append(1.0)
        if new_frame.input_tensor is not None:
            future_gradient = self._kernel.predict_future_state_gradient(new_frame)
            new_frame.future_gradient_prediction = future_gradient
            if future_gradient is not None:
                assert not tf.reduce_any(tf.math.is_nan(future_gradient))
                state_gradients.append(future_gradient)
                state_gradient_weights.append(self._config.future_gradient_coefficient)

        # Combine the gathered gradients and update the frame.
        if state_gradients:
            if len(state_gradients) == 1:
                combined_gradient = state_gradients[0]
            else:
                # noinspection PyTypeChecker
                gradient_total = tf.add_n([weight * gradient for weight, gradient
                                           in zip(state_gradient_weights, state_gradients)])
                if self._config.stabilized_gradient:
                    weight_total = sum(state_gradient_weights)
                    combined_gradient = gradient_total / weight_total
                else:
                    combined_gradient = gradient_total
            assert not tf.reduce_any(tf.math.is_nan(combined_gradient))
            self._previous_frame.current_state_gradient = combined_gradient
