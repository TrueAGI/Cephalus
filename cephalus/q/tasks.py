from typing import Callable, Any, Tuple, Union, Optional, TYPE_CHECKING

import tensorflow as tf

from cephalus.modules.interface import StateKernelModule

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel
    from cephalus.q.td import TDAgent


class RewardDrivenTask(StateKernelModule):

    recent_reward_update_rate: float = 0.01
    recent_episode_duration_update_rate: float = 0.1

    def __init__(self, agent: 'TDAgent',
                 objective: Callable[[Any], Tuple[Union[float, tf.Tensor], bool]]):
        super().__init__()

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

    def configure(self, kernel: 'StateKernel') -> None:
        super().configure(kernel)

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
        self.recent_reward += (reward - self.recent_reward) * max(update_rate,
                                                                  self.recent_reward_update_rate)
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

        # TODO: Use logging.
        print("Total steps for task %s: %s" %
              (self.agent.__class__.__name__, self.total_reward_samples))
        print("Reward for task %s: %s" % (self.agent.__class__.__name__, reward))
        print("Mean reward for task: %s: %s" % (self.agent.__class__.__name__, self.mean_reward))
        print("Recent reward for task: %s: %s" %
              (self.agent.__class__.__name__, self.recent_reward))
        print("Current episode steps for task %s: %s" %
              (self.agent.__class__.__name__, self.episodic_reward_samples))
        print("Episode count for task %s: %s" %
              (self.agent.__class__.__name__, self.episode_count))
        print("Mean episode reward for task %s: %s" %
              (self.agent.__class__.__name__, self.mean_episodic_reward))
        print("Mean episode duration for task %s: %s" %
              (self.agent.__class__.__name__, self.mean_episode_duration))
        print("Recent episode duration for task %s: %s" %
              (self.agent.__class__.__name__, self.recent_episode_duration))
        print("Longest episode duration for task %s: %s" %
              (self.agent.__class__.__name__, self.longest_episode))

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
