"""The state kernel is a configurable kernel for online neural learning of sequential state updates.
This module defines the state kernel and the abstract interfaces for its modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any, Iterable, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from cephalus.support import StandardizedTensorShape


@dataclass
class StateKernelConfig:
    """The configuration data for a state kernel. Includes basic information necessary for the
    kernel and its modules to construct, train, and utilize their neural models."""

    # The width of the state space. The state space is always 1-dimensional.
    state_width: int

    # The shape of the input space.
    input_shape: StandardizedTensorShape

    # A model template with a structure suitable for predicting the next state. The model must
    # accept two inputs, one for previous state and one for current input, in that order. It must
    # return a single output, for the current state.
    model_template: Model

    # The optimizer to be used for all the models.
    optimizer: Optimizer

    # Additional weights defined outside the kernel that should be trained using the kernel's
    # state prediction loss.
    additional_trainable_weights: Optional[Tuple[tf.Variable, ...]]


@dataclass
class StateKernelFrame:
    """A collection of all the information collected about a state prediction which may be required
    to train the kernel. Kernels may subclass this and add their own fields."""

    previous_state: Union[tf.Tensor, tf.Variable]
    tape: Optional[tf.GradientTape]

    input_tensor: tf.Tensor = None
    current_state: Optional[tf.Tensor] = None
    current_state_gradient: Optional[tf.Tensor] = None
    future_gradient_prediction: Optional[tf.Tensor] = None
    trained: bool = False

    module_data: Dict['StateKernelModule', Dict[str, Any]] = field(default_factory=dict)


class StateKernel:
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
    _modules: List['StateKernelModule']
    _state_prediction_provider: 'StatePredictionProvider' = None
    _gradient_prediction_provider: 'GradientPredictionProvider' = None
    _trainable_weights: Tuple[tf.Variable, ...] = None

    def __init__(self, modules: Iterable['StateKernelModule'] = None,
                 config: Optional[StateKernelConfig] = None):
        self._modules = []
        if modules:
            for module in modules:
                self.add_module(module)
        if config is not None:
            self.configure(config)

    def add_module(self, module: 'StateKernelModule') -> None:
        """Add a module to the state kernel. The kernel's modules must be added before the kernel is
        configured."""
        assert self._config is None
        self._modules.append(module)

    def configure(self, config: StateKernelConfig) -> None:
        """Configure the state kernel and its modules for a particular environment. The kernel must
        be configured after any modules are added and before the kernel is used."""
        assert self._config is None, "Kernel is already configured."

        # Apply the configuration
        self._config = config
        for module in self._modules:
            module.configure(self)

        # Ensure invariants and constraints are respected.
        if self._state_prediction_provider is None:
            from cephalus.modules.state_prediction import StandardStatePredictionProvider
            module = StandardStatePredictionProvider()
            self._modules.append(module)
            module.configure(self)
        if self.initial_state is None:
            self.initial_state = tf.Variable(tf.zeros(config.state_width), name='initial_state')

        # Gather the trainable weights in advance to avoid having to rebuild the list repeatedly.
        weights = []
        if self.initial_state_is_trainable:
            weights.append(self.initial_state)
        for module in self._modules:
            weights.extend(module.get_trainable_weights())
        self._trainable_weights = tuple(weights)

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
    def input_shape(self) -> StandardizedTensorShape:
        """The shape of the input tensors accepted by this state kernel."""
        assert self._config is not None
        return self._config.input_shape

    @property
    def optimizer(self) -> Optional[Optimizer]:
        """The optimizer used by the state kernel and its modules to optimize their neural
        models."""
        assert self._config is not None
        return self._config.optimizer

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
    def initial_state(self) -> Union[tf.Tensor, tf.Variable]:
        """The initial state tensor or variable used by the state kernel at the beginning of a
        state stream."""
        assert self._initial_state is not None
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: Union[tf.Tensor, tf.Variable]) -> None:
        """The initial state tensor or variable used by the state kernel at the beginning of a
        state stream."""
        self._initial_state = value
        self._initial_state_is_trainable = isinstance(value, tf.Variable)

    @property
    def initial_state_is_trainable(self) -> bool:
        """This property's value indicates whether the initial state is a trainable variable -- as
        opposed to a non-trainable tensor."""
        assert self._initial_state is not None
        return self._initial_state_is_trainable

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        """Return a tuple of the trainable weights of the neural models used by the state kernel and
        its modules."""
        assert self._config is not None
        assert self._trainable_weights is not None
        return self._trainable_weights

    def get_loss(self, decision_frame: StateKernelFrame) -> Optional[tf.Tensor]:
        """Return the computed loss for the neural models used by the state kernel and its modules.
        Values which have already been computed (i.e. the current state) will not be recomputed. Use
        the provided gradient tape in the decision frame to get the gradients of the parameters."""
        assert self._config is not None
        assert decision_frame.current_state is not None

        losses = []
        for module in self._modules:
            module_loss = module.get_loss(decision_frame)
            if module_loss is not None:
                # noinspection PyTypeChecker
                losses.append(module.loss_scale * module_loss)

        # TODO: Do we want to allow scaling of state and gradient prediction losses computed here?
        if decision_frame.current_state_gradient is not None:
            state_prediction = decision_frame.current_state
            state_target = tf.stop_gradient(state_prediction +
                                            decision_frame.current_state_gradient)
            state_loss = tf.reduce_sum(tf.square(state_target - state_prediction))
            losses.append(state_loss)

            gradient_prediction = decision_frame.future_gradient_prediction
            if gradient_prediction is not None:
                gradient_target = decision_frame.current_state_gradient
                gradient_loss = tf.reduce_sum(tf.square(gradient_target - gradient_prediction))
                losses.append(gradient_loss)

        if len(losses) == 1:
            return losses[0]
        elif losses:
            return tf.add_n(losses)
        else:
            return None

    def get_previous_state(self, previous_frame: StateKernelFrame = None) -> Union[tf.Tensor,
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

    def new_decision_frame(self, previous_frame: StateKernelFrame = None) -> StateKernelFrame:
        """Return a new decision frame, initializing the required fields appropriately."""
        assert self._config is not None
        new_frame = StateKernelFrame(
            previous_state=self.get_previous_state(previous_frame),
            tape=self.new_gradient_tape(),
        )
        for module in self._modules:
            module_data = module.get_new_decision_frame_data(new_frame, previous_frame)
            if module_data is not None:
                new_frame.module_data[module] = module_data
        return new_frame

    def accept_input(self, frame: StateKernelFrame, input_tensor: tf.Tensor) -> None:
        """Accept a new input tensor. Record the input tensor in the frame, along with the state
        prediction made from it."""
        assert self._config is not None
        assert frame.current_state is None
        assert frame.input_tensor is None
        frame.input_tensor = input_tensor
        self.predict_state(frame)

    def predict_state(self, frame: StateKernelFrame) -> None:
        """Predict the current state from the previous state and the current input. Record the
        prediction in the frame."""
        assert self._config is not None
        assert frame.input_tensor is not None
        assert frame.current_state is None
        assert self._state_prediction_provider is not None
        new_state = self._state_prediction_provider.predict_state(frame)
        assert new_state is not None
        assert not tf.reduce_any(tf.math.is_nan(new_state))
        new_state = tf.clip_by_value(new_state, -1e6, 1e6)
        frame.current_state = new_state

    def predict_future_state_gradient(self, frame: StateKernelFrame) -> Optional[tf.Tensor]:
        """Predict the expected gradient of the previous state w.r.t. the current state's loss."""
        assert self._config is not None
        if self._gradient_prediction_provider is None:
            return None
        return self._gradient_prediction_provider.predict_future_state_gradient(frame)

    def train(self, decision_frame: StateKernelFrame) -> None:
        """Train the neural models used by the state kernel and its modules."""
        assert self._config is not None
        assert not decision_frame.trained
        assert decision_frame.tape is not None

        try:
            if decision_frame.current_state_gradient is None:
                return  # Nothing to do.

            weights = self.get_trainable_weights()
            if not weights:
                return  # Nothing to do.

            loss = self.get_loss(decision_frame)
            if loss is None:
                return  # Nothing to do.

            tf.assert_equal(tf.size(loss), 1)
            while tf.rank(loss) > 0:
                loss = loss[0]
            assert not tf.reduce_any(tf.math.is_nan(loss))

            loss_gradients = decision_frame.tape.gradient(loss, weights)
            assert not any(tf.reduce_any(tf.math.is_nan(loss_gradient))
                           for loss_gradient in loss_gradients
                           if loss_gradient is not None)
            loss_gradients, _ = tf.clip_by_global_norm(loss_gradients, 1.0)
            self.optimizer.apply_gradients(zip(loss_gradients, weights))
        finally:
            decision_frame.tape.__exit__(None, None, None)
            decision_frame.tape = None
            decision_frame.trained = True


class StateKernelModule(ABC):
    """A pluggable module for the state kernel."""

    _kernel: Optional[StateKernel] = None
    _loss_scale: float = 1.0

    @abstractmethod
    def configure(self, kernel: StateKernel) -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required."""
        assert self._kernel is None, "Kernel module is already configured."
        self._kernel = kernel

    @abstractmethod
    def get_trainable_weights(self) -> List[tf.Variable]:
        """Return a list of the trainable weights of the primary and any secondary models."""
        raise NotImplementedError()

    @abstractmethod
    def get_loss(self, decision_frame: StateKernelFrame) -> Optional[tf.Tensor]:
        """Return the computed loss for any models, or None if there are no trainable weights.
        Values which have already been computed (i.e. the current state) will not be recomputed. Use
        the provided gradient tape in the decision frame to get the gradients of the parameters.

        The returned loss should not be scaled. The kernel will apply loss scaling at a later
        point."""
        raise NotImplementedError()

    @property
    def kernel(self) -> Optional[StateKernel]:
        """The state kernel this module is configured for."""
        return self._kernel

    @property
    def config(self) -> Optional[StateKernelConfig]:
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
    def input_shape(self) -> StandardizedTensorShape:
        """The shape of the input space for the state kernel."""
        assert self._kernel is not None
        return self._kernel.input_shape

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

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def get_new_decision_frame_data(self, frame: StateKernelFrame,
                                    previous_frame: StateKernelFrame = None) \
            -> Optional[Dict[str, Any]]:
        """Return any additional initialization information specific to the module that must be
        stored in the frame. The module's data will be stored in frame.module_data[module]."""
        return None


class StatePredictionProvider(StateKernelModule, ABC):
    """A state kernel module which provides state predictions for its kernel. Only one state
    prediction provider can be configured for a given kernel."""

    @abstractmethod
    def configure(self, kernel: StateKernel) -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required. Notify the kernel that this module will be its state prediction
        provider."""
        super().configure(kernel)
        kernel.state_prediction_provider = self

    @abstractmethod
    def predict_state(self, decision_frame: StateKernelFrame) -> Optional[tf.Tensor]:
        """Predict the current state from the previous state and the current input."""
        raise NotImplementedError()


class GradientPredictionProvider(StateKernelModule, ABC):
    """A state kernel module which provides gradient predictions for its kernel. Only one gradient
    prediction provider can be configured for a given kernel."""

    @abstractmethod
    def configure(self, kernel: StateKernel) -> None:
        """Configure the module to work with a configured state kernel, building any neural models
        that are required. Notify the kernel that this module will be its gradient prediction
        provider."""
        super().configure(kernel)
        kernel.gradient_prediction_provider = self

    @abstractmethod
    def predict_future_state_gradient(self, frame: StateKernelFrame) -> Optional[tf.Tensor]:
        """Predict the gradient for the previous state given the current state."""
        raise NotImplementedError()
