"""The state kernel is a configurable kernel for online neural learning of sequential state updates.
This module defines the state kernel and the abstract interfaces for its modules."""

from typing import Optional, Union, Iterable, Tuple, Set, TypeVar, \
    Generic

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from cephalus.config import StateKernelConfig
from cephalus.frame import StateFrame
from cephalus.modules.interface import StateKernelModule, StatePredictionProvider, \
    GradientPredictionProvider, GradientProvider, InputProvider, InputAttentionProvider

__all__ = [
    'StateKernel'
]


Environment = TypeVar('Environment')


class StateKernel(Generic[Environment]):
    """A configurable kernel for online neural learning of sequential state updates.

    The state kernel is initialized by adding one or more kernel modules, each of which provides
    some sort of individualized functionality such as state prediction, future gradient prediction,
    or additional losses which help the kernel learn a better state representation. These modules
    can be mixed and matched to optimize the kernel for a particular environment. Once the modules
    have been added, a call is made to kernel.configure(), passing it the kernel configuration data,
    which includes information specific to the environment such as the input shape and the suggested
    model architecture for learning the state transition space.

    After the state kernel is initialized and configured, it is passed, along with an environment
    instance, to a harness, which acts as a coordinator of the interactions between the kernel and
    the environment."""

    _config: Optional[StateKernelConfig] = None
    _initial_state: Union[tf.Tensor, tf.Variable] = None
    _initial_state_is_trainable: bool = False
    _default_input: Union[tf.Tensor, tf.Variable] = None
    _default_input_is_trainable: bool = False
    _modules: Set['StateKernelModule[Environment]']
    _input_attention_provider: 'InputAttentionProvider' = None
    _state_prediction_provider: 'StatePredictionProvider' = None
    _gradient_prediction_provider: 'GradientPredictionProvider' = None
    _trainable_weights: Tuple[tf.Variable, ...] = None

    def __init__(self, modules: Iterable['StateKernelModule[Environment]'] = None,
                 config: Optional[StateKernelConfig] = None):
        self._modules = set()
        if modules:
            for module in modules:
                self.add_module(module)
        if config is not None:
            self.configure(config)

    def add_module(self, module: 'StateKernelModule[Environment]') -> None:
        """Add a module to the state kernel."""
        if module not in self._modules:
            self._modules.add(module)
            if self._config:
                module.configure(self)
                self.recompute_trainable_weights()

    def discard_module(self, module: 'StateKernelModule[Environment]') -> None:
        """Remove a module from the state kernel."""
        assert module is not self._state_prediction_provider
        if module in self._modules:
            self._modules.remove(module)
            if self._config:
                self.recompute_trainable_weights()

    def configure(self, config: StateKernelConfig) -> None:
        """Configure the state kernel and its modules for a particular environment. The kernel must
        be configured after any modules are added and before the kernel is used."""
        assert self._config is None, "Kernel is already configured."

        # Apply the configuration
        self._config = config
        for module in self._modules:
            module.configure(self)

        # Ensure invariants and constraints are respected.
        if self._input_attention_provider is None:
            from cephalus.modules.input_attention import StandardInputAttentionProvider
            module = StandardInputAttentionProvider()
            self.add_module(module)
        if self._state_prediction_provider is None:
            from cephalus.modules.state_prediction import StandardStatePredictionProvider
            module = StandardStatePredictionProvider()
            self.add_module(module)
        if self._initial_state is None:
            self.initial_state = tf.Variable(tf.zeros(config.state_width), name='initial_state')

        self.recompute_trainable_weights()

    def step(self, environment: Environment, previous_frame: StateFrame = None) -> StateFrame:
        """Run the kernel in the environment for a single step. Return the new frame."""
        frame = self.new_frame(previous_frame)
        self.gather_inputs(environment, frame)
        self.input_attention_provider.attend_inputs(frame)
        self.predict_state(frame)

        if previous_frame is not None:
            self.update_previous_frame(previous_frame, frame)
            # We train even if there were no external gradients, as some modules have
            # internally-induced gradients.
            self.train(previous_frame)

        # Ask the tasks for the current state's gradients and record them for later training.
        state_gradients = []
        for module in self._modules:
            if not isinstance(module, GradientProvider):
                continue
            state_gradient = module.get_current_state_gradient(environment, frame)
            if state_gradient is not None:
                assert not tf.reduce_any(tf.math.is_nan(state_gradient))
                state_gradients.append(state_gradient)
        if state_gradients:
            combined_gradient = tf.add_n(state_gradients) / len(state_gradients)
            frame.current_state_gradient = combined_gradient

        return frame

    @property
    def config(self) -> Optional[StateKernelConfig]:
        """The state kernel's configuration."""
        return self._config

    @property
    def state_width(self) -> int:
        """The width of the state tensors used by this state kernel."""
        assert self._config is not None
        return self._config.state_width

    @property
    def input_width(self) -> int:
        """The width of the input tensors accepted by this state kernel."""
        assert self._config is not None
        return self._config.input_width

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """The optimizer used by the state kernel and its modules to optimize their neural
        models."""
        assert self._config is not None
        return self._config.optimizer

    @property
    def future_gradient_coefficient(self) -> float:
        """The coefficient to multiply predicted gradients for future states that have been
        propagated back to the current state. Acts as a future discount rate."""
        return self._config.future_gradient_coefficient

    @property
    def stabilized_gradient(self) -> bool:
        """Whether to divide the sum of the current true gradient and the discounted future gradient
        by (1 + future_gradient_coefficient). If true, effectively normalizes their sum to the
        natural range of the true gradient."""
        return self._config.stabilized_gradient

    @property
    def state_prediction_provider(self) -> Optional['StatePredictionProvider']:
        """The module which is designated as the state kernel's state prediction provider."""
        return self._state_prediction_provider

    @state_prediction_provider.setter
    def state_prediction_provider(self, module: 'StatePredictionProvider') -> None:
        """The module which is designated as the state kernel's state prediction provider."""
        assert self._state_prediction_provider is None
        self._state_prediction_provider = module

    @property
    def gradient_prediction_provider(self) -> Optional['GradientPredictionProvider']:
        """The module which is designated as the state kernel's state gradient prediction
        provider. The state gradient prediction provider provides a prediction of the previous
        state's gradient w.r.t. the current state's loss. This is conceptually analogous to the
        future reward predictions provided by the next state/action pair to train the previous
        state/action pair in reinforcement learning algorithms such as SARSA or Q-Learning."""
        return self._gradient_prediction_provider

    @gradient_prediction_provider.setter
    def gradient_prediction_provider(self, module: 'GradientPredictionProvider') -> None:
        """The module which is designated as the state kernel's state gradient prediction
        provider. The state gradient prediction provider provides a prediction of the previous
        state's gradient w.r.t. the current state's loss. This is conceptually analogous to the
        future reward predictions provided by the next state/action pair to train the previous
        state/action pair in reinforcement learning algorithms such as SARSA or Q-Learning."""
        assert self._gradient_prediction_provider is None
        self._gradient_prediction_provider = module

    @property
    def input_attention_provider(self) -> Optional['InputAttentionProvider']:
        return self._input_attention_provider

    @input_attention_provider.setter
    def input_attention_provider(self, value: 'InputAttentionProvider') -> None:
        self._input_attention_provider = value

    @property
    def initial_state(self) -> Union[tf.Tensor, tf.Variable]:
        """The initial state tensor or variable used by the state kernel at the beginning of a
        state stream."""
        assert self._initial_state is not None
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: Union[tf.Tensor, tf.Variable]) -> None:
        """The initial state tensor or variable used by the state kernel at the beginning of a
        state stream."""
        assert self._initial_state is None
        self._initial_state = value
        self._initial_state_is_trainable = isinstance(value, tf.Variable)

    @property
    def initial_state_is_trainable(self) -> bool:
        """This property's value indicates whether the initial state is a trainable variable -- as
        opposed to a non-trainable tensor."""
        assert self._initial_state is not None
        return self._initial_state_is_trainable

    @property
    def default_input(self) -> Union[tf.Tensor, tf.Variable]:
        """The default input tensor or variable used by the state kernel when there are no
        inputs."""
        assert self._default_input is not None
        return self._default_input

    @default_input.setter
    def default_input(self, value: Union[tf.Tensor, tf.Variable]) -> None:
        """The default input tensor or variable used by the state kernel when there are no
        inputs."""
        self._default_input = value
        self._default_input_is_trainable = isinstance(value, tf.Variable)

    @property
    def default_input_is_trainable(self) -> bool:
        """This property's value indicates whether the default input is a trainable variable -- as
        opposed to a non-trainable tensor."""
        assert self._default_input is not None
        return self._default_input_is_trainable

    def recompute_trainable_weights(self):
        """Recompute the trainable weights after a configuration or module change."""
        assert self._config
        weights = []
        if self.initial_state_is_trainable:
            weights.append(self.initial_state)
        if self.default_input_is_trainable:
            weights.append(self.default_input)
        for module in self._modules:
            weights.extend(module.get_trainable_weights())
        self._trainable_weights = tuple(weights)

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        """Return a tuple of the trainable weights of the neural models used by the state kernel and
        its modules."""
        assert self._config is not None
        assert self._trainable_weights is not None
        return self._trainable_weights

    def get_loss(self, frame: StateFrame) -> Optional[tf.Tensor]:
        """Return the computed loss for the neural models used by the state kernel and its modules.
        Values which have already been computed (i.e. the current state) will not be recomputed. Use
        the provided gradient tape in the decision frame to get the gradients of the parameters."""
        assert self._config is not None
        assert frame.current_state is not None

        losses = []
        for module in self._modules:
            module_loss = module.get_loss(frame)
            if module_loss is not None:
                # noinspection PyTypeChecker
                losses.append(module.loss_scale * module_loss)

        # TODO: Do we want to allow scaling of state and gradient prediction losses computed here?
        if frame.current_state_gradient is not None:
            state_prediction = frame.current_state
            state_target = tf.stop_gradient(state_prediction +
                                            frame.current_state_gradient)
            state_loss = tf.reduce_sum(tf.square(state_target - state_prediction))
            losses.append(state_loss)

            gradient_prediction = frame.future_gradient_prediction
            if gradient_prediction is not None:
                gradient_target = frame.current_state_gradient
                gradient_loss = tf.reduce_sum(tf.square(gradient_target - gradient_prediction))
                losses.append(gradient_loss)

        if len(losses) == 1:
            return losses[0]
        elif losses:
            return tf.add_n(losses)
        else:
            return None

    def get_previous_state(self, previous_frame: StateFrame = None) -> Union[tf.Tensor,
                                                                             tf.Variable]:
        """Return the previous state tensor or variable given the previous frame. If no previous
        frame is provided, default to the initial state."""
        assert self._config is not None
        if previous_frame is None:
            return self.initial_state
        previous_state = previous_frame.current_state
        assert previous_state is not None
        return previous_state

    def new_gradient_tape(self) -> Optional[tf.GradientTape]:
        """Return a new gradient tape, which is already open and set to record the trainable
         weights."""
        assert self._config is not None
        tape = tf.GradientTape(persistent=True, watch_accessed_variables=False)
        tape.__enter__()  # TODO: Is there a cleaner way to do this?
        for weight in self.get_trainable_weights():
            tape.watch(weight)
        return tape

    def new_frame(self, previous_frame: StateFrame = None) -> StateFrame:
        """Return a new decision frame, initializing the required fields appropriately."""
        assert self._config is not None
        new_frame = StateFrame(
            previous_state=self.get_previous_state(previous_frame),
            tape=self.new_gradient_tape(),
        )
        for module in self._modules:
            module_data = module.get_new_frame_data(new_frame, previous_frame)
            if module_data is not None:
                new_frame.module_data[module] = module_data
        return new_frame

    def gather_inputs(self, environment: Environment, frame: StateFrame) -> None:
        """Gather new input tensors from the environment. Record the input tensors in the frame."""
        assert self._config is not None
        assert frame.current_state is None
        assert frame.input_tensors is None
        input_tensors = [self.default_input]
        for module in self._modules:
            if not isinstance(module, InputProvider):
                continue
            input_tensor = module.get_input(environment, frame)
            if input_tensor is not None:
                assert input_tensor.shape == (self.input_width,)
                input_tensors.append(input_tensor)
        frame.input_tensors = input_tensors

    def predict_state(self, frame: StateFrame) -> None:
        """Predict the current state from the previous state and the current input. Record the
        prediction in the frame."""
        assert self._config is not None
        assert frame.input_tensors is not None
        assert frame.current_state is None
        assert self._state_prediction_provider is not None
        new_state = self._state_prediction_provider.predict_state(frame)
        assert new_state is not None
        assert not tf.reduce_any(tf.math.is_nan(new_state))
        new_state = tf.clip_by_value(new_state, -1e6, 1e6)
        frame.current_state = new_state

    def predict_future_state_gradient(self, frame: StateFrame) -> Optional[tf.Tensor]:
        """Predict the expected gradient of the previous state w.r.t. the current state's loss."""
        assert self._config is not None
        if self._gradient_prediction_provider is None:
            return None
        return self._gradient_prediction_provider.predict_future_state_gradient(frame)

    def update_previous_frame(self, previous_frame: StateFrame,
                              new_frame: StateFrame) -> None:
        """Update the previous frame's state gradient with the gradient from the current frame."""
        assert previous_frame is not None
        # Gather gradients for the previous state.
        state_gradients = []
        state_gradient_weights = []
        if previous_frame.current_state_gradient is not None:
            state_gradients.append(previous_frame.current_state_gradient)
            state_gradient_weights.append(1.0)
        future_gradient = self.predict_future_state_gradient(new_frame)
        new_frame.future_gradient_prediction = future_gradient
        if future_gradient is not None:
            assert not tf.reduce_any(tf.math.is_nan(future_gradient))
            state_gradients.append(future_gradient)
            state_gradient_weights.append(self.future_gradient_coefficient)

        # Combine the gathered gradients and update the frame.
        if state_gradients:
            if len(state_gradients) == 1:
                combined_gradient = state_gradients[0]
            else:
                # noinspection PyTypeChecker
                gradient_total = tf.add_n([weight * gradient for weight, gradient
                                           in zip(state_gradient_weights, state_gradients)])
                if self.stabilized_gradient:
                    weight_total = sum(state_gradient_weights)
                    combined_gradient = gradient_total / weight_total
                else:
                    combined_gradient = gradient_total
            assert not tf.reduce_any(tf.math.is_nan(combined_gradient))
            previous_frame.current_state_gradient = combined_gradient

    def train(self, frame: StateFrame) -> None:
        """Train the neural models used by the state kernel and its modules."""
        assert self._config is not None
        assert not frame.trained
        assert frame.tape is not None

        try:
            if frame.current_state_gradient is None:
                return  # Nothing to do.

            weights = self.get_trainable_weights()
            if not weights:
                return  # Nothing to do.

            loss = self.get_loss(frame)
            if loss is None:
                return  # Nothing to do.

            tf.assert_equal(tf.size(loss), 1)
            while tf.rank(loss) > 0:
                loss = loss[0]
            assert not tf.reduce_any(tf.math.is_nan(loss))

            loss_gradients = frame.tape.gradient(loss, weights)
            assert not any(tf.reduce_any(tf.math.is_nan(loss_gradient))
                           for loss_gradient in loss_gradients
                           if loss_gradient is not None)
            loss_gradients, _ = tf.clip_by_global_norm(loss_gradients, 1.0)
            self.optimizer.apply_gradients(zip(loss_gradients, weights))
        finally:
            frame.tape.__exit__(None, None, None)
            frame.tape = None
            frame.trained = True
