from typing import List, Union, Tuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions as tfd

from cephalus.config import StateKernelConfig
from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel
from cephalus.modules.sensing import sensor
from cephalus.q.action_policies import DiscreteActionPolicy
from cephalus.q.probabilistic_models import ProbabilisticModel
from cephalus.q.tasks import RewardDrivenTask
from cephalus.q.td import TDAgent
from cephalus.stream import StateStream


class GymEnvSolver:

    def __init__(self, env: gym.Env, kernel: StateKernel, agent: TDAgent, visualize: bool = False):
        self.env = env
        self.kernel = kernel
        self.agent = agent
        self.visualize = visualize

        self.observation = env.reset()
        self.reset = True

        self.modules = [
            self.sense_observation,
            self.sense_reset,
            RewardDrivenTask(agent, self.objective)
        ]

        kernel.add_modules(*self.modules)

    def __del__(self) -> None:
        self.close()

    def close(self):
        self.env.close()
        self.kernel.discard_modules(*self.modules)

    @sensor
    def sense_observation(self, _frame: StateFrame) -> np.ndarray:
        return self.observation

    @sensor
    def sense_reset(self, _frame: StateFrame) -> float:
        return float(self.reset)

    def objective(self, action: tf.Tensor) -> Tuple[float, bool]:
        if self.visualize:
            self.env.render()
        self.observation, reward, self.reset, _ = self.env.step(action)
        if self.reset:
            self.env.reset()
        return reward, self.reset


def test_cartpole():
    state_width = 10
    input_width = 4

    state_in = Input(state_width, name='state_in')
    input_in = Input(input_width, name='input_in')
    concat = Concatenate()([state_in, input_in])
    hidden = Dense(state_width + input_width, activation='tanh')(concat)
    state_out = Dense(state_width)(hidden)
    state_model = Model([state_in, input_in], state_out, name='state_update')

    kernel = StateKernel()
    kernel.configure(
        StateKernelConfig(
            state_width=state_width,
            input_width=input_width,
            model_template=state_model,
            optimizer=Adam(),
            future_gradient_coefficient=0.99,
            stabilized_gradient=True
        )
    )

    env = gym.make('CartPole-v0')

    # Structure borrowed from https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/ and
    # adapted to probability distributions.
    q_model_body = Sequential([
        Dense(24, input_dim=4, activation='tanh'),
        Dense(48, activation='tanh'),
    ])

    state = Input(kernel.input_width, name='state')
    shared = q_model_body(state)
    q_mean = Dense(2)(shared)
    q_stddev = Dense(2)(shared)
    q_model = Model(state, [q_mean, q_stddev])

    class ConditionalNormalDistribution(ProbabilisticModel):

        def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
            mean, stddev = self.parameter_model(inputs)
            return tfd.Normal(loc=mean, scale=stddev)

    q_dist = ConditionalNormalDistribution(q_model)

    # We don't use epsilon greedy because the q value distribution is probabilistic, which means it
    # will handle exploration all on its own. As the distribution converges, the variance in the
    # predicted q values will decline, causing the exploration rate at a natural rate dictated by
    # the agent's needs. Search 'Thompson sampling' for an idea of how this works.
    action_policy = DiscreteActionPolicy()

    agent = TDAgent(
        q_dist,
        action_policy,
        .99,
        stabilize=True
    )

    solver = GymEnvSolver(
        env,
        kernel,
        agent,
        visualize=True
    )

    # Create and run the state stream.
    stream = StateStream(kernel)
    stream.run()

    solver.close()
