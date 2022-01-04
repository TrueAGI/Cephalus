from typing import Callable, Any, Tuple, Union, Optional, TYPE_CHECKING

import tensorflow as tf

from cephalus.modules.interface import StateKernelModule

if TYPE_CHECKING:
    from cephalus.frame import StateFrame
    from cephalus.kernel import StateKernel
    from cephalus.q.td import TDAgent


class RewardDrivenTask(StateKernelModule):

    def __init__(self, agent: 'TDAgent',
                 objective: Callable[[Any], Tuple[Union[float, tf.Tensor], bool]]):
        self.agent = agent
        self.objective = objective

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
        loss = self.agent.accept_reward(reward)
        if reset:
            loss += self.agent.reset()
        return loss
