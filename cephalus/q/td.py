from typing import Union, Optional, TYPE_CHECKING, Tuple

import tensorflow as tf

from cephalus.modeled import Modeled
from cephalus.q.action_policies import ActionPolicy, ActionDecision

if TYPE_CHECKING:
    from cephalus.q.probabilistic_models import ProbabilisticModel


class TDAgent(Modeled):

    def __init__(self, q_model: 'ProbabilisticModel', action_policy: ActionPolicy,
                 discount: float, stabilize: bool = True):
        self._q_model = q_model
        self._action_policy = action_policy
        self.discount = discount
        self.stabilize = stabilize

        self._previous_decision = None
        self._current_decision = None

    @property
    def q_model(self) -> 'ProbabilisticModel':
        return self._q_model

    @property
    def action_policy(self) -> ActionPolicy:
        return self._action_policy

    def build(self) -> None:
        self._q_model.build()
        self._action_policy.build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return self._q_model.get_trainable_weights() + self._action_policy.get_trainable_weights()

    def _update_previous_decision(self) -> None:
        if not self._previous_decision:
            return None

        assert self._previous_decision.reward is not None
        assert self._previous_decision.q_value is None

        if self._current_decision:
            assert self._current_decision.reward is None
            # noinspection PyTypeChecker
            previous_q_value = (self._previous_decision.reward +
                                self.discount * self._current_decision.q_value_prediction)
            if self.stabilize:
                previous_q_value /= (1.0 + self.discount)
        else:
            previous_q_value = self._previous_decision.reward

        self._previous_decision.q_value = previous_q_value

    def _close_previous_decision(self) -> Optional[tf.Tensor]:
        if self._previous_decision:
            loss = self._action_policy.get_loss(self._previous_decision)
        else:
            loss = None
        self._previous_decision = self._current_decision
        self._current_decision = None
        return loss

    def reset(self) -> Optional[tf.Tensor]:
        self._update_previous_decision()
        return self._close_previous_decision()

    def choose_action(self, state_input: tf.Tensor):
        assert self._current_decision is None

        step = self._previous_decision.step + 1 if self._previous_decision else 0
        decision = ActionDecision(state_input, self.q_model, step)
        self.action_policy.choose_action(decision)
        self.action_policy.predict_q_value(decision)
        self._current_decision = decision

        self._update_previous_decision()

        return decision.action

    def accept_reward(self, reward: Union[float, tf.Tensor]) -> Optional[tf.Tensor]:
        assert self._current_decision is not None
        assert self._current_decision.reward is None

        self._current_decision.reward = reward
        return self._close_previous_decision()
