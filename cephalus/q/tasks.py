import logging
from typing import Callable, Any, Tuple, Union, Optional, TYPE_CHECKING

import tensorflow as tf

from cephalus.modules.interface import StateKernelModule

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel
    from cephalus.q.td import TDAgent


LOGGER = logging.getLogger(__name__)


class RewardDrivenTask(StateKernelModule):

    recent_reward_update_rate: float = 0.01
    recent_episode_duration_update_rate: float = 0.1

    def __init__(self, agent: 'TDAgent',
                 objective: Callable[[Any], Tuple[Union[float, tf.Tensor], bool]], *,
                 name: str = None):
        self.agent = agent
        self.objective = objective

        self.total_reward_samples = 0
        self.mean_reward = 0.0
        self.recent_reward = 0.0
        self.episodic_reward_samples = 0
        self.mean_episodic_reward = 0.0
        self.mean_episode_duration = 0.0
        self.recent_episode_duration = 0.0
        self.longest_episode = 0
        self.episode_count = 0

        super().__init__(name=name)

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)
        self.agent.configure(kernel)

    def build(self) -> None:
        self.agent.build()
        super().build()

    def get_trainable_weights(self) -> Tuple[tf.Variable, ...]:
        return self.agent.get_trainable_weights()

    def get_loss(self, previous_frame: 'StateFrame',
                 current_frame: 'StateFrame') -> Optional[tf.Tensor]:
        action = self.agent.choose_action(current_frame.current_state)
        reward, reset = self.objective(action)

        # TODO: This section should go in separate modules that can be added to the task
        #       compositionally.
        self.total_reward_samples += 1
        update_rate = 1 / self.total_reward_samples
        self.mean_reward += (reward - self.mean_reward) * update_rate
        self.recent_reward += ((reward - self.recent_reward) *
                               max(update_rate, self.recent_reward_update_rate))
        if reset:
            self.episode_count += 1
            update_rate = 1 / self.episode_count
            self.mean_episode_duration += (
                (self.episodic_reward_samples - self.mean_episode_duration) *
                update_rate
            )
            self.recent_episode_duration += (
                (self.episodic_reward_samples - self.recent_episode_duration) *
                max(update_rate, self.recent_episode_duration_update_rate)
            )
            self.longest_episode = max(self.longest_episode, self.episodic_reward_samples)
            self.episodic_reward_samples = 1
            self.mean_episodic_reward = reward
        else:
            self.episodic_reward_samples += 1
            self.longest_episode = max(self.longest_episode, self.episodic_reward_samples)
            update_rate = 1 / self.episodic_reward_samples
            self.mean_episodic_reward += (reward - self.mean_episodic_reward) * update_rate
            if self.episode_count == 0:
                self.mean_episode_duration = self.episodic_reward_samples

        if LOGGER.isEnabledFor(logging.INFO):
            LOGGER.info("Total steps for agent %s: %s", self.agent, self.total_reward_samples)
            LOGGER.info("Reward for agent %s: %s", self.agent, reward)
            LOGGER.info("Mean reward for agent: %s: %s", self.agent, self.mean_reward)
            LOGGER.info("Recent reward for agent: %s: %s", self.agent, self.recent_reward)
            LOGGER.info("Current episode steps for agent %s: %s", self.agent,
                        self.episodic_reward_samples)
            LOGGER.info("Episode count for agent %s: %s", self.agent, self.episode_count)
            LOGGER.info("Mean episode reward for agent %s: %s", self.agent,
                        self.mean_episodic_reward)
            LOGGER.info("Mean episode duration for agent %s: %s", self.agent,
                        self.mean_episode_duration)
            LOGGER.info("Recent episode duration for agent %s: %s", self.agent,
                        self.recent_episode_duration)
            LOGGER.info("Longest episode duration for agent %s: %s", self.agent,
                        self.longest_episode)

        losses = [self.agent.accept_reward(reward)]
        if reset:
            losses.append(self.agent.reset())
        loss_sum = None
        for loss in losses:
            if loss_sum is None:
                loss_sum = loss
            elif loss is not None:
                loss_sum += loss
        return loss_sum
