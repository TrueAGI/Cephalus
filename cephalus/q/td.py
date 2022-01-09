import logging
from typing import Union, Optional, TYPE_CHECKING, Tuple, Callable

import tensorflow as tf

from cephalus.modeled import Modeled
from cephalus.modules.interface import StateKernelModule
from cephalus.q.action_policies import ActionPolicy, ActionDecision

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel
    from cephalus.q.probabilistic_models import ProbabilisticModel
    from cephalus.q.doubt_estimator import DoubtEstimator


LOGGER = logging.getLogger(__name__)


class TDAgent(StateKernelModule):
    max_observable_reward: float = None
    min_observable_reward: float = None
    episodes: int = 0
    total_steps: int = 0

    def __init__(self, q_model: 'ProbabilisticModel', action_policy: ActionPolicy,
                 discount: Union[float, Callable[[float, float], float]],
                 stabilize: bool = True, doubt_estimator: 'DoubtEstimator' = None, *,
                 name: str = None):
        self._q_model: 'ProbabilisticModel' = q_model
        self._action_policy = action_policy
        self.discount: float = discount
        self.stabilize: bool = stabilize
        self._doubt_estimator: 'DoubtEstimator' = doubt_estimator
        self._previous_decision: Optional[ActionDecision] = None
        self._current_decision: Optional[ActionDecision] = None

        super().__init__(name=name)

    @property
    def q_model(self) -> 'ProbabilisticModel':
        return self._q_model

    @property
    def doubt_estimator(self) -> Optional['DoubtEstimator']:
        return self._doubt_estimator

    @property
    def action_policy(self) -> ActionPolicy:
        return self._action_policy

    def clone(self) -> 'TDAgent':
        return type(self)(self._q_model, self._action_policy, self.discount, self.stabilize,
                          self._doubt_estimator)

    def get_discount(self) -> float:
        if callable(self.discount):
            return self.discount(self.episodes, self.total_steps)
        else:
            assert isinstance(self.discount, float)
            return self.discount

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        if isinstance(self._doubt_estimator, StateKernelModule):
            self._doubt_estimator.configure(kernel)

    def build(self) -> None:
        self._q_model.build()
        self._action_policy.build()
        if isinstance(self._doubt_estimator, Modeled):
            self._doubt_estimator.build()

        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        result = self._q_model.get_trainable_weights() + self._action_policy.get_trainable_weights()
        if isinstance(self._doubt_estimator, StateKernelModule):
            result += self._doubt_estimator.get_trainable_weights()
        return result

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        if isinstance(self._doubt_estimator, StateKernelModule):
            return self._doubt_estimator.get_loss(previous_frame, current_frame)
        else:
            return None

    def _update_previous_decision(self) -> None:
        if not self._previous_decision:
            return None

        assert self._previous_decision.reward is not None
        assert self._previous_decision.q_value_target is None

        if self._current_decision:
            assert self._current_decision.reward is None
            # noinspection PyTypeChecker
            prediction = float(self._current_decision.exploit_q_value_prediction)
            discount = self.get_discount()

            if self.stabilize:
                # Sometimes the model can generalize poorly, resulting in predicted values that are
                # outside the bounds of what is actually reasonable.
                if self.min_observable_reward is not None:
                    prediction = max(prediction, self.min_observable_reward)
                if self.max_observable_reward is not None:
                    prediction = min(prediction, self.max_observable_reward)
                previous_q_value = ((self._previous_decision.reward + discount * prediction) /
                                    (1.0 + discount))
            else:
                # noinspection PyTypeChecker
                previous_q_value = self._previous_decision.reward + discount * prediction
        else:
            if self.stabilize:
                discount = self.get_discount()
                previous_q_value = self._previous_decision.reward / (1.0 + discount)
            else:
                previous_q_value = self._previous_decision.reward

        self._previous_decision.q_value_target = previous_q_value

    def _close_previous_decision(self) -> Optional[tf.Tensor]:
        if self._previous_decision:
            LOGGER.info("Previous step's predicted Q-value for task %s: %s", self,
                        self._previous_decision.selected_q_value_prediction.numpy())
            # noinspection PyTypeChecker
            LOGGER.info("Previous step's target Q-value for task %s: %s", self,
                        float(self._previous_decision.q_value_target))
            policy_loss = self._action_policy.get_loss(self._previous_decision)
            if self._previous_decision:
                previous_doubt = self._previous_decision.doubt
            else:
                previous_doubt = 0.0
            if self._current_decision:
                current_doubt = self._current_decision.doubt
            else:
                current_doubt = 0.0
            previous_doubt += 0.000000001  # Numerical stability
            current_doubt += 0.000000001
            doubt_ratio = 2.0 * (previous_doubt / (previous_doubt + current_doubt))
            assert doubt_ratio > 0.0
            # noinspection PyTypeChecker
            loss = policy_loss * doubt_ratio
        else:
            policy_loss = None
            loss = None
        self._previous_decision = self._current_decision
        self._current_decision = None
        if self._doubt_estimator:
            if policy_loss is None:
                policy_loss = 0.0
            if self._previous_decision:
                doubt_loss = self._doubt_estimator.get_loss(self._previous_decision,
                                                            float(policy_loss))
            else:
                doubt_loss = None
            if loss is None:
                loss = doubt_loss
            elif doubt_loss is not None:
                loss += doubt_loss
        return loss

    def reset(self) -> Optional[tf.Tensor]:
        self.episodes += 1
        self._update_previous_decision()

        # We have to treat a reset as an observation of reward zero, because the bounds we're
        # establishing apply to q value predictions, not rewards, and we use a q value prediction
        # of zero for end-of-episode.
        if self.stabilize:
            reward = 0.0

            if self.max_observable_reward is None:
                self.max_observable_reward = reward
            else:
                self.max_observable_reward = max(reward, self.max_observable_reward)
            if self.min_observable_reward is None:
                self.min_observable_reward = reward
            else:
                self.min_observable_reward = min(reward, self.min_observable_reward)

            LOGGER.info("Max observable reward for task %s: %s", self, self.max_observable_reward)
            LOGGER.info("Min observable reward for task %s: %s", self, self.min_observable_reward)

        return self._close_previous_decision()

    def choose_action(self, state_input: tf.Tensor):
        assert self._current_decision is None

        step = self._previous_decision.step + 1 if self._previous_decision else 0
        decision = ActionDecision(state_input, self.q_model, step)
        if self._doubt_estimator:
            decision.doubt = float(self._doubt_estimator.get_doubt(decision))
        self.action_policy.choose_action(decision)
        self._current_decision = decision

        self._update_previous_decision()

        return decision.selected_action

    def accept_reward(self, reward: Union[float, tf.Tensor]) -> Optional[tf.Tensor]:
        assert self._current_decision is not None
        assert self._current_decision.reward is None

        self.total_steps += 1

        self._current_decision.reward = reward

        if self.stabilize:
            if self.max_observable_reward is None:
                self.max_observable_reward = reward
            else:
                self.max_observable_reward = max(reward, self.max_observable_reward)
            if self.min_observable_reward is None:
                self.min_observable_reward = reward
            else:
                self.min_observable_reward = min(reward, self.min_observable_reward)

            LOGGER.info("Max observable reward for task %s: %s", self, self.max_observable_reward)
            LOGGER.info("Min observable reward for task %s: %s", self, self.min_observable_reward)

        return self._close_previous_decision()
