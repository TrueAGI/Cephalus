from typing import Tuple, Optional

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import SGD

from cephalus.config import StateKernelConfig
from cephalus.frame import StateFrame
from cephalus.kernel import StateKernel
from cephalus.modules.autoencoder import StateAutoencoder
from cephalus.modules.input_prediction import InputPrediction
from cephalus.modules.sensing import sensor
from cephalus.q.action_policies import DiscreteActionPolicy
from cephalus.q.epsilon_greedy import EpsilonGreedy
from cephalus.q.probabilistic_models import DeterministicModel
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
        self.last_action = env.action_space.sample()
        self.reward = 0.0

        self.modules = [
            # The agent can sense the environment directly.
            sensor(env.observation_space.shape, GymEnvSolver.sense_observation),

            # The agent can sense the action it last took.
            sensor(env.action_space.shape, GymEnvSolver.sense_action),

            # The agent can sense the reward it last received.
            GymEnvSolver.sense_reward,

            # The agent can sense when the environment is reset.
            GymEnvSolver.sense_reset,

            # The agent attempts to maximize the reward objective.
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

    def sense_action(self, _frame: StateFrame) -> np.ndarray:
        return self.last_action

    @sensor((1,))
    def sense_reset(self, _frame: StateFrame) -> float:
        return float(self.reset)

    @sensor((1,))
    def sense_reward(self, _frame: StateFrame) -> float:
        return float(self.reward)

    def objective(self, action: tf.Tensor) -> Tuple[float, bool]:
        if self.visualize:
            self.env.render()
        self.observation, self.reward, self.reset, _ = self.env.step(action.numpy())
        self.last_action = action
        if self.reset:
            self.env.reset()
        return self.reward, self.reset


def build_kernel() -> StateKernel:
    state_width = 10
    input_width = 10

    print("Building state model.")
    state_model = Sequential(
        [
            Dense(state_width + input_width, activation='tanh'),
            Dense(state_width, activation='tanh')
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
        )
    )
    kernel.add_modules(StateAutoencoder())
    kernel.add_modules(InputPrediction())
    # kernel.add_modules(LossStateTD(loss_scale=0.99))

    return kernel


def build_cartpole_agent(state_width: int) -> TDAgent:
    print("Building q model.")

    state_input = Input(state_width, name='state_input')
    shared = Sequential([
        Dense(24, input_dim=state_width, activation='tanh'),
        Dense(48, activation='tanh')
    ])(state_input)
    state_value = Dense(1, use_bias=True, activation='linear')(shared)
    relative_action_value = Dense(2, use_bias=True, activation='linear')(shared)
    q_mean = Lambda(lambda x: tf.repeat(x[0], 2, axis=-1) + x[1] - tf.reduce_max(x[1]))(
        [state_value, relative_action_value]
    )
    q_model = Model(state_input, q_mean)
    q_dist = DeterministicModel(q_model)

    # q_stddev = Dense(2, activation='softplus')(shared)
    # q_model = Model(state_input, [q_mean, q_stddev])
    # q_dist = ProbabilisticModel(q_model, tfd.Normal)

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

    return agent


def test_single(steps: Optional[int] = 10):
    """Test a single kernel running a single agent on a single stream.

    NOTE: This is not a performance test. We don't care if the environment
    is actually learned or solved, just that we don't get exceptions."""

    kernel = build_kernel()
    agent = build_cartpole_agent(state_width=kernel.state_width)

    print("Creating environment.")
    env = gym.make('CartPole-v0')
    solver = GymEnvSolver(
        env,
        kernel,
        agent,
        visualize=True
    )

    print("Creating and running the state stream.")
    stream = StateStream(kernel, environment=solver)
    stream.run(steps=steps)

    print("Closing the solver.")
    solver.close()

    print("Done.")


def test_multiple_streams(steps: Optional[int] = 10, stream_count: int = 2):
    """Test a single kernel running a single agent on multiple concurrent streams.

    NOTE: This is not a performance test. We don't care if the environment
    is actually learned or solved, just that we don't get exceptions."""

    kernel = build_kernel()
    agent = build_cartpole_agent(state_width=kernel.state_width)

    print("Creating the state streams.")
    streams = []
    for _ in range(stream_count):
        env = gym.make('CartPole-v0')
        solver = GymEnvSolver(
            env,
            kernel,
            agent.clone(),
            visualize=True
        )
        stream = StateStream(kernel, environment=solver)
        streams.append(stream)

    print("Running the state streams.")
    # TODO: This isn't true concurrency. For that, we will need to do some locking in the kernel.
    if steps is None:
        while True:
            for stream in streams:
                stream.step()
    else:
        for step in range(steps):
            for stream in streams:
                stream.step()

    for stream in streams:
        print("Closing the solvers.")
        stream.environment.close()

    print("Done.")


def test_multiple_agents(steps: Optional[int] = 10, agent_count: int = 2):
    """Test a single kernel running multiple agents on a single stream.

    NOTE: This is not a performance test. We don't care if the environment
    is actually learned or solved, just that we don't get exceptions."""
    assert False  # TODO


def test_multiple(steps: Optional[int] = 10, agent_count: int = 2):
    """Test a single kernel running multiple agents on multiple streams.

    NOTE: This is not a performance test. We don't care if the environment
    is actually learned or solved, just that we don't get exceptions."""
    assert False  # TODO
