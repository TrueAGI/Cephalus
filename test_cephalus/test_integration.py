from typing import List, Union, Tuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow_probability import distributions as tfd

from cephalus.config import StateKernelConfig
from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel
from cephalus.modules.autoencoder import StateAutoencoder
from cephalus.modules.retroactive_loss import LossStateTD
from cephalus.modules.sensing import sensor
from cephalus.q.action_policies import DiscreteActionPolicy
from cephalus.q.epsilon_greedy import EpsilonGreedy
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
            sensor(env.observation_space.shape, GymEnvSolver.sense_observation),
            GymEnvSolver.sense_reset,
            RewardDrivenTask(agent, self.objective)
        ]

        kernel.add_modules(*self.modules)

    def __del__(self) -> None:
        self.close()

    def close(self):
        self.env.close()
        self.kernel.discard_modules(*self.modules)

    def sense_observation(self, _frame: StateFrame) -> np.ndarray:
        return self.observation

    @sensor((1,))
    def sense_reset(self, _frame: StateFrame) -> float:
        return float(self.reset)

    def objective(self, action: tf.Tensor) -> Tuple[float, bool]:
        # TODO: The agent should also be able to sense what action it took.
        if self.visualize:
            self.env.render()
        self.observation, reward, self.reset, _ = self.env.step(action.numpy())
        if self.reset:
            self.env.reset()
        return reward, self.reset


def test_cartpole(steps: int = 1000):
    state_width = 10
    input_width = 4

    print("Building state model.")
    state_model = Sequential(
        [
            Dense(state_width + input_width, activation='tanh'),
            Dense(state_width)
        ], name='state_update'
    )

    print("Creating kernel.")
    kernel = StateKernel()
    kernel.configure(
        StateKernelConfig(
            state_width=state_width,
            input_width=input_width,
            model_template=state_model,
            optimizer=SGD(),
            future_gradient_coefficient=0.99,
            stabilized_gradient=True
        )
    )
    kernel.add_modules(LossStateTD())
    kernel.add_modules(StateAutoencoder())

    print("Building q model.")

    # Structure borrowed from https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/ and
    # adapted to probability distributions.
    q_model_body = Sequential([
        Dense(24, input_dim=state_width, activation='tanh'),
        Dense(48, activation='tanh'),
    ])

    state_input = Input(kernel.state_width, name='state_input')
    shared = q_model_body(state_input)
    q_mean = Dense(2)(shared)
    q_stddev = Dense(2, activation='softplus')(shared)
    q_model = Model(state_input, [q_mean, q_stddev])

    class ConditionalNormalDistribution(ProbabilisticModel):

        def __call__(self, inputs: Union[tf.Tensor, List[tf.Tensor]]) -> tfd.Distribution:
            mean, stddev = self.parameter_model(inputs)
            return tfd.Normal(loc=mean, scale=stddev)

    q_dist = ConditionalNormalDistribution(q_model)

    print("Creating action policy.")
    action_policy = DiscreteActionPolicy(
        exploration_policy=EpsilonGreedy()
    )

    print("Creating agent.")
    agent = TDAgent(
        q_dist,
        action_policy,
        .99,
        stabilize=True
    )

    print("Creating environment.")
    env = gym.make('CartPole-v0')

    print("Creating solver.")
    solver = GymEnvSolver(
        env,
        kernel,
        agent,
        visualize=True
    )

    print("Creating and running the state stream.")
    # Create and run the state stream.
    stream = StateStream(kernel, environment=solver)
    stream.run(steps=steps)

    print("Closing the solver.")
    solver.close()

    print("Done.")
