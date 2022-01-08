"""The state kernel is a configurable kernel for online neural learning of sequential state updates.
This module defines the state kernel and the abstract interfaces for its modules."""

from typing import Optional, Union, Iterable, Tuple, Set, TypeVar, \
    Generic, List

import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer

from cephalus.config import StateKernelConfig
from cephalus.frame import StateFrame
from cephalus.modeled import Modeled
from cephalus.modules.interface import StateKernelModule, StatePredictionProvider, \
    RetroactiveLossProvider, Sensor, InputAttentionProvider, InputSample

__all__ = [
    'StateKernel'
]


Environment = TypeVar('Environment')


class StateKernel(Modeled, Generic[Environment]):
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
    _modules: Set['StateKernelModule[Environment]']
    _input_attention_provider: 'InputAttentionProvider' = None
    _state_prediction_provider: 'StatePredictionProvider' = None
    _retroactive_gradient_provider: 'RetroactiveLossProvider' = None
    _trainable_weights: Tuple[tf.Variable, ...] = None

    def __init__(self, modules: Iterable['StateKernelModule[Environment]'] = None,
                 config: Optional[StateKernelConfig] = None):
        self._modules = set()
        if modules:
            self.add_modules(*modules)
        if config is not None:
            self.configure(config)

    def add_modules(self, *modules: 'StateKernelModule[Environment]') -> None:
        """Add a module to the state kernel."""
        for module in modules:
            if module not in self._modules:
                self._modules.add(module)
                if self._config:
                    module.configure(self)
                    if self._built:
                        module.build()
        if self.is_built:
            self.recompute_trainable_weights()

    def discard_modules(self, *modules: 'StateKernelModule[Environment]') -> None:
        """Remove a module from the state kernel."""
        for module in modules:
            assert module is not self._state_prediction_provider
            if module in self._modules:
                self._modules.remove(module)
        if self.is_built:
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
            self.add_modules(module)
        if self._state_prediction_provider is None:
            from cephalus.modules.state_prediction import StandardStatePredictionProvider
            module = StandardStatePredictionProvider()
            self.add_modules(module)
        if self._initial_state is None:
            self.initial_state = tf.Variable(tf.zeros(config.state_width), name='initial_state')

    def build(self):
        assert self._config is not None
        for module in self._modules:
            module.build()
        self.recompute_trainable_weights()
        super().build()

    def step(self, environment: Environment, previous_frame: StateFrame = None) -> StateFrame:
        """Run the kernel in the environment for a single step. Return the new frame."""
        if not self.is_built:
            self.build()

        frame = self.new_frame(previous_frame)

        self.gather_inputs(environment, frame)
        self.input_attention_provider.attend_inputs(frame)
        self.predict_state(frame)

        if previous_frame is not None:
            # We train even if there were no external gradients, as some modules have
            # internally-induced gradients.
            self.train(previous_frame, frame)

        return frame

    @property
    def config(self) -> Optional[StateKernelConfig]:
        """The state kernel's configuration."""
        return self._config

    @property
    def dtype(self) -> tf.DType:
        """The data type used by the kernel."""
        return self._config.dtype

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
    def sensor_embedding_width(self) -> int:
        """The width of the sensor embedding width accepted by this state kernel."""
        assert self._config is not None
        return self._config.input_width

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
    def retroactive_gradient_provider(self) -> Optional['RetroactiveLossProvider']:
        """The module which is designated as the state kernel's retroactive state gradient
        provider. The retroactive state gradient provider augments the state gradient with
        additional components based on its predicted effect on the next state. This is conceptually
        analogous to the Q values provided by the next state/action pair to train the previous
        state/action pair in reinforcement learning algorithms such as SARSA or Q-Learning."""
        return self._retroactive_gradient_provider

    @retroactive_gradient_provider.setter
    def retroactive_gradient_provider(self, module: 'RetroactiveLossProvider') -> None:
        """The module which is designated as the state kernel's retroactive state gradient
        provider. The retroactive state gradient provider augments the state gradient with
        additional components based on its predicted effect on the next state. This is conceptually
        analogous to the Q values provided by the next state/action pair to train the previous
        state/action pair in reinforcement learning algorithms such as SARSA or Q-Learning."""
        assert self._retroactive_gradient_provider is None
        self._retroactive_gradient_provider = module

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

    def recompute_trainable_weights(self):
        """Recompute the trainable weights after a configuration or module change."""
        assert self._config
        weights = []
        if self._initial_state_is_trainable:
            assert self.initial_state.dtype == self.dtype
            weights.append(self.initial_state)
        for module in self._modules:
            module_weights = module.get_trainable_weights()
            for weight in module_weights:
                assert weight.dtype == self.dtype, (module, weight.name)
            weights.extend(module_weights)
        self._trainable_weights = tuple(weights)

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        """Return a tuple of the trainable weights of the neural models used by the state kernel and
        its modules."""
        assert self._config is not None
        assert self._trainable_weights is not None
        return self._trainable_weights

    def get_loss(self, previous_frame: StateFrame,
                 current_frame: StateFrame) -> Optional[tf.Tensor]:
        """Return the computed loss for the previous frame's state tensor. Values which have already
        been computed for the frame (i.e. the current state) will not be recomputed. Use the
        provided gradient tape in the decision frame to get the gradients of the parameters."""
        assert self._config is not None
        assert previous_frame.current_state is not None

        losses: List[tf.Tensor] = []
        for module in self._modules:
            module_loss = module.get_loss(previous_frame, current_frame)
            if module_loss is not None and module.loss_scale > 0.0:
                assert module_loss.shape == (), "Invalid loss shape returned from %r" % module
                # noinspection PyTypeChecker
                scaled_module_loss: tf.Tensor = module.loss_scale * module_loss
                losses.append(scaled_module_loss)

                # TODO: Use logging. Also, provide a way for the user to capture loss curves over
                #       time for each module and retrieve them easily. And modules should be named
                #       (manually, or else automatically at config time with a reasonable default)
                #       so we can identify them easily.
                print("Loss for %s: %s" % (module.__class__.__name__, scaled_module_loss.numpy()))

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
            clock_ticks=0 if previous_frame is None else 1 + previous_frame.clock_ticks
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
        assert frame.input_samples is None
        samples: List[InputSample] = []
        for module in self._modules:
            if not isinstance(module, Sensor):
                continue
            for sample in module.get_inputs(environment, frame):
                assert sample.value.shape == (self.input_width,)
                samples.append(sample)
        frame.input_samples = samples

    def predict_state(self, frame: StateFrame) -> None:
        """Predict the current state from the previous state and the current input. Record the
        prediction in the frame."""
        assert self._config is not None
        assert frame.input_samples is not None
        assert frame.current_state is None
        assert self._state_prediction_provider is not None
        new_state = self._state_prediction_provider.predict_state(frame)
        assert new_state is not None
        assert not tf.reduce_any(tf.math.is_nan(new_state))
        new_state = tf.clip_by_value(new_state, -1e6, 1e6)
        frame.current_state = new_state

    def train(self, previous_frame: StateFrame, current_frame: StateFrame) -> None:
        """Train the neural models used by the state kernel and its modules."""
        assert self._config is not None
        assert not previous_frame.trained
        assert previous_frame.tape is not None

        try:
            loss = self.get_loss(previous_frame, current_frame)
            assert loss.dtype == self.dtype
            if loss is None:
                loss = tf.zeros(())
        finally:
            previous_frame.tape.__exit__(None, None, None)

        tf.assert_rank(loss, 0)
        assert not tf.math.is_nan(loss)

        weights = self.get_trainable_weights()
        loss_gradients = previous_frame.tape.gradient(loss, weights)
        assert not any(tf.reduce_any(tf.math.is_nan(loss_gradient))
                       for loss_gradient in loss_gradients
                       if loss_gradient is not None)
        loss_gradients, _ = tf.clip_by_global_norm(loss_gradients, 1.0)
        gradient_weight_pairs = [(gradient, weight)
                                 for gradient, weight in zip(loss_gradients, weights)
                                 if gradient is not None]
        if gradient_weight_pairs:
            self.optimizer.apply_gradients(gradient_weight_pairs)

        # Train the loss providers here, before we remove the tape, in case they need gradient
        # information.
        previous_frame.combined_loss = loss
        for module in self._modules:
            if isinstance(module, RetroactiveLossProvider):
                module.train_retroactive_loss(previous_frame, current_frame)

        previous_frame.tape = None
        previous_frame.trained = True
