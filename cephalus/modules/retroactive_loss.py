# TODO: This is where all the interesting stuff goes. Look at rltd_gp.hand_coded_solutions and
#       'Temporal Differences for Infinite-Horizon Backpropagation.md' in the RLTD project, and
#       the various approaches evaluated in the related 'GradientEstimation' and
#       'RewardInducedGradients' projects for ideas. Also take a look at the 'RLAgents' project for
#       general design ideas. Any existing policy gradient approach can also be adapted for state
#       representation induction by treating negative prediction loss as reward, so make a list of
#       solidly performant policy gradient algorithms and implement them here. And finally, methods
#       for 'cheating', i.e. concatenated input histories used in conjunction with LSTMs or
#       attention mechanisms should also be implemented here, using the module-specific data stored
#       in StateFrame to capture histories.
# TODO: Once the different approaches are coded here, we'll need to run some
#       tests to compare their performance.

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import clone_model

from cephalus.modules.interface import RetroactiveLossProvider

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel


# NOTE: This class reimplements the method defined in rig.methods.backpropagation in the
#       RewardInducedGradients project.
class LossStateTD(RetroactiveLossProvider):
    """The Temporal Differences-based approach applied direction at the level of losses.

    Loss TD works by predicting a loss for the preceding state, and simply returning that loss.
    Kernel training will backpropagate the gradient through the models used to predict the loss and
    move the state in a direction that reduces loss.
    """

    # NOTE: The discount rate is controlled by the generic module parameter, 'loss_scale'.

    _loss_model: Model = None

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self._loss_model = Sequential(clone_model(kernel.config.model_template).layers[:-1] +
                                      [Dense(1, 'softplus')])
        # We use SGD irrespective of which optimizer the kernel is configured to use, because SGD
        # provides more stable estimates of the input gradients that other training methods. This
        # model's sole purpose is to provide input gradients to train the state model with; the
        # actual loss estimates it computes are useless in and of themselves.
        self._loss_model.compile('sgd', 'mse')

    def train_retroactive_loss(self, previous_frame: StateFrame, current_frame: StateFrame) -> None:
        assert previous_frame.current_state is not None
        assert previous_frame.attended_input_tensor is not None
        assert previous_frame.combined_loss is not None
        state = previous_frame.current_state[tf.newaxis, :]
        input_tensor = previous_frame.attended_input_tensor[tf.newaxis, :]
        loss_target = previous_frame.combined_loss[tf.newaxis, tf.newaxis]
        self._loss_model.fit([state, input_tensor], loss_target)

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        assert previous_frame.current_state is not None
        assert previous_frame.attended_input_tensor is not None
        assert previous_frame.combined_loss is None
        state = previous_frame.current_state[tf.newaxis, :]
        input_tensor = previous_frame.attended_input_tensor[tf.newaxis, :]
        # NOTE: Thanks to this line, we can't return the loss model's weights in
        #       get_trainable_weights(). If we did, our loss model would be trained to minimize its
        #       *prediction* of loss, irrespective of what the actual loss would be. That's not what
        #       we want at all. We want the weights of other modules which actually contribute to
        #       the computation of the state tensor to move in a direction that minimizes loss,
        #       while the loss model defined here only provides the gradients to them, while
        #       striving to provide an accurate estimate of loss.
        return self._loss_model([state, input_tensor])


# TODO: The autoencoder-based methods defined in rig.methods.decoder and rig.methods.encoder in the
#       RewardInducedGradients project should be subclassed from this class.
class TargetStateTD(RetroactiveLossProvider, ABC):
    """Abstract base class for Temporal Differences-based approaches applied to predecessor state
    targets.

    Target TD works by computing a preceding state that probably would have improved the Q value,
    and using that as a training target for the preceding state.
    """

    # NOTE: The discount rate is controlled by the generic module parameter, 'loss_scale'.

    @abstractmethod
    def compute_previous_state_target(self, previous_frame: 'StateFrame',
                                      current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        raise NotImplementedError()

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        # Compute the target.
        previous_state_target = self.compute_previous_state_target(previous_frame, current_frame)

        # Convert the target to a loss.
        loss = tf.reduce_mean(0.5 * tf.square(tf.stop_gradient(previous_state_target) -
                                              previous_frame.current_state))

        # Return the loss.
        return loss


# TODO: The gradient-based methods outlined in 'Temporal Differences for Infinite-Horizon
#       Backpropagation.md' and defined in rltd_gp.hand_coded_solutions, both in the RLTD project,
#       should be implemented as concrete subclasses of this class.
class GradientStateTD(RetroactiveLossProvider, ABC):
    """Abstract base class for Temporal Differences-based approaches applied to predecessor state
    gradients.

    Gradient TD works by computing a gradient for the preceding state that would probably improve
    the Q value, and using that as a training gradient for the preceding state.
    """

    # NOTE: The discount rate is controlled by the generic module parameter, 'loss_scale'.

    @abstractmethod
    def compute_previous_state_gradient(self, previous_frame: 'StateFrame',
                                        current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        """Compute the partial of the previous state's gradient w.r.t. the predicted combined
        gradient of the current state. The new frame's state and input tensors will be populated,
        but its gradient information will not be. The computation this method performs is analogous
        to determining the value of `Q(s_(t+1), a_(t+1))` in SARSA/Q-learning, but for gradients
        instead of rewards. However, what we are actually computing is
        `d/s_t Q(s_(t+1), a_(t+1)) = d/s_t Q(S(s_t), A(s_t)) =
        d/dS Q(S(s_t), A(s_t)) dS/ds_t + d/dA Q(S(s_t), A(s_t)) dA/ds_t`, or an estimate thereof."""
        raise NotImplementedError()

    @abstractmethod
    def train_retroactive_loss(self, previous_frame: 'StateFrame',
                               current_frame: 'StateFrame') -> None:
        """Train the combined gradient prediction model for the current state. The new frame will be
        populated with the combined gradient for the state. This method should train the model to
        predict the combined gradient from the input and state tensors. This is analogous to
        updating `Q(s_t, a_t) <- r_t + discount * Q(s_(t+1), a_(t+1))` in SARSA or
        `Q(s_t, a_t) <- r_t + discount * max(Q(s_(t+1), a) for a in A)` in Q-learning. However,
        the model we are training does not compute `Q`, but rather its derivative w.r.t. `s_t`:
        `d/s_t (r_t + discount * Q(s_(t+1), a_(t+1))) =
        d/s_t (-L(s_t) + discount * Q(s_(t+1), a_(t+1))) =
        d/s_t (-L(s_t) + discount * Q(S(s_t), A(s_t))) =
        -d/s_t L(s_t) + discount * (d/dS Q(S(s_t), A(s_t)) dS/ds_t +
                                    d/dA Q(S(s_t), A(s_t)) dA/ds_t)`
        where `L(s_t)` is the total task-defined loss as a function of `s_t` gathered during time
        step t.
        """
        raise NotImplementedError()

    def compute_retroactive_loss(self, previous_frame: 'StateFrame',
                                 current_frame: 'StateFrame') -> tf.Tensor:
        """Compute and return the previous frame's combined state gradient. The result returned
        is analogous to `r_t + discount * Q(s_(t+1), a_(t+1))` in SARSA or
        `r_t + discount * max(Q(s_(t+1), a) for a in A)` in Q-learning. However, what we are
        actually computing (or estimating) is `d/s_t (r_t + discount * Q(s_(t+1), a_(t+1))) =
        d/s_t (-L(s_t) + discount * Q(s_(t+1), a_(t+1))) =
        d/s_t (-L(s_t) + discount * Q(S(s_t), A(s_t))) =
        -d/s_t L(s_t) + discount * (d/dS Q(S(s_t), A(s_t)) dS/ds_t +
                                    d/dA Q(S(s_t), A(s_t)) dA/ds_t)`
        where `L(s_t)` is the total task-defined loss as a function of `s_t`.
        """
        assert current_frame.current_state is not None

        # Compute the gradient.
        gradient = self.compute_previous_state_gradient(previous_frame, current_frame)

        # Convert the gradient to a target.
        target = tf.stop_gradient(previous_frame.current_state - gradient)

        # Convert the target to a loss.
        loss = tf.reduce_sum(0.5 * tf.square(target - previous_frame.current_state))

        # Return the loss.
        return loss
