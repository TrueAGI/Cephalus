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

    doubt: float = 0.0

    action_distribution: tfd.Distribution = None
    selected_q_value_distribution: tfd.Distribution = None

    selected_action: tf.Tensor = None
    selected_q_value_prediction: tf.Tensor = None

    exploit_action: tf.Tensor = None
    exploit_q_value_prediction: tf.Tensor = None

    reward: Union[float, tf.Tensor] = None
    q_value_target: tf.Tensor = None


class ActionPolicy(Modeled, ABC):

    @abstractmethod
    def choose_action(self, decision: ActionDecision) -> None:
        raise NotImplementedError()

    def get_loss(self, decision: ActionDecision) -> tf.Tensor:
        loss = decision.q_model.distribution_loss(
            decision.selected_q_value_distribution,
            tf.stop_gradient(decision.q_value_target)
        )
        return tf.clip_by_norm(loss, 1.0)


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

        selected_action = decision.action_distribution.sample()

        q_value_distribution = decision.q_model([
            decision.state,
            tf.stop_gradient(decision.selected_action)
        ])

        selected_mean = q_value_distribution.mean()

        decision.selected_action = selected_action
        decision.selected_q_value_prediction = selected_mean

        decision.exploit_action = selected_action
        decision.exploit_q_value_prediction = selected_mean

        decision.selected_q_value_prediction = q_value_distribution

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
        # NOTE: I was taking index 0 of the joint distribution here, to remove the batch axis that
        # Keras requires, but I had to change it to remove the batch axis in the various places
        # where it's used, below, instead. It turns out there's a bug in tensorflow_probability
        # that causes an index error if you index iteratively instead of all at once. It's reported
        # here: https://github.com/tensorflow/probability/issues/1487
        joint_q_value_distribution = decision.q_model(decision.state[tf.newaxis, :])

        q_value_predictions = joint_q_value_distribution.sample()[0]
        best_action = tf.argmax(q_value_predictions)
        if self.exploration_policy is None or not self.exploration_policy(decision.step):
            selected_action = best_action
        else:
            selected_action = tf.random.uniform((), 0, q_value_predictions.shape[-1],
                                                dtype=tf.int64)

        selected_mean = joint_q_value_distribution.mean()[0, selected_action]
        exploit_mean = joint_q_value_distribution.mean()[0, best_action]

        decision.selected_action = selected_action
        decision.selected_q_value_prediction = selected_mean

        decision.exploit_action = best_action
        decision.exploit_q_value_prediction = exploit_mean

        decision.selected_q_value_distribution = \
            joint_q_value_distribution[0, decision.selected_action]
