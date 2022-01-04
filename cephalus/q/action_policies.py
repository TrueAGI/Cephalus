from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Callable, TYPE_CHECKING, Tuple

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from cephalus.modeled import Modeled

if TYPE_CHECKING:
    from cephalus.q.probabilistic_models import ProbabilisticModel


@dataclass
class ActionDecision:
    state: tf.Tensor
    q_model: 'ProbabilisticModel'
    step: int

    action: tf.Tensor = None
    q_value_prediction: tf.Tensor = None
    reward: Union[float, tf.Tensor] = None
    q_value: tf.Tensor = None

    # Bookkeeping fields, used by the action policies to store information they need to pass between
    # methods for efficiency.
    action_distribution: tfd.Distribution = None
    joint_q_value_distribution: tfd.Distribution = None
    q_value_distribution: tfd.Distribution = None


class ActionPolicy(Modeled, ABC):

    @abstractmethod
    def choose_action(self, decision: ActionDecision) -> None:
        raise NotImplementedError()

    @staticmethod
    def predict_q_value(decision: ActionDecision) -> None:
        decision.q_value_prediction = decision.q_value_distribution.mean()

    def get_loss(self, decision: ActionDecision) -> tf.Tensor:
        return -decision.q_value_distribution.log_prob(decision.q_value)


# TODO: Implement REINFORCE, Q Actor/Critic, AAC, PPO, etc., as well as the novel algorithms
#       defined in the RewardInducedGradients project.
class ContinuousActionPolicy(ActionPolicy, ABC):

    def __init__(self, policy_model: 'ProbabilisticModel'):
        self.policy_model = policy_model

    @abstractmethod
    def get_policy_loss(self, decision: ActionDecision) -> tf.Tensor:
        raise NotImplementedError()

    def build(self) -> None:
        self.policy_model.build()
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return self.policy_model.get_trainable_weights()

    def choose_action(self, decision: ActionDecision) -> None:
        decision.action_distribution = self.policy_model(decision.state)
        decision.action = decision.action_distribution.sample()
        decision.q_value_distribution = decision.q_model([decision.state,
                                                          tf.stop_gradient(decision.action)])

    def get_loss(self, decision: ActionDecision) -> tf.Tensor:
        return super().get_loss(decision) + self.get_policy_loss(decision)


class DiscreteActionPolicy(ActionPolicy):

    def __init__(self, exploration_policy: Callable[[int], bool] = None):
        self.exploration_policy = exploration_policy

    def build(self) -> None:
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return ()

    def choose_action(self, decision: ActionDecision) -> None:
        decision.joint_q_value_distribution = decision.q_model(decision.state[tf.newaxis, :])[0]
        q_values = decision.joint_q_value_distribution.sample()
        if self.exploration_policy is None or not self.exploration_policy(decision.step):
            decision.action = tf.argmax(q_values)
        else:
            decision.action = tf.random.uniform((), 0, q_values.shape[-1], dtype=tf.int64)
        decision.q_value_distribution = decision.joint_q_value_distribution[decision.action]
