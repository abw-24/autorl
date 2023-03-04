
from gym import spaces
import random
import numpy as np
from abc import ABCMeta, abstractmethod


class GymAgent(object, metaclass=ABCMeta):

    def __init__(self, env, discount=0.99, config=None):
        """
        Base RL agent for Gym RL environments. Does some initial introspection
        to set the state and action space info needed for model configuration
        and sets up generics + abstract methods to be overwritten by children.
        :param env: Gym environment
        :param discount: Reward discounts
        :param config: Generic dictionary configuration (for e.g. model
            hyperparameter configuration)
        """

        self._env = env
        self._discount = discount
        self._state_space = env.observation_space
        self._action_space = env.action_space
        self._config = config
        self._train_max_reward = 0.0
        self._replay_buffer = []
        self._buffer_size = None
        self._batch_size = None

        # Check the environment, set shapes and types
        assert isinstance(self._state_space, spaces.Box), \
            "Expecting a Gym `Box` state space."

        self._state_shape = list(self._state_space.shape)

        n_dim = len(self._state_shape)
        if n_dim == 1:
            self._state_type = "vector"
        else:
            self._state_type = "{d}-tensor".format(d=n_dim)

        if isinstance(self._action_space, spaces.discrete.Discrete):
            self._action_shape = int(self._action_space.n)
            self._action_type = "discrete"
        else:
            raise NotImplementedError(
                "Currently only supporting `Discrete` action spaces."
            )

    def reset(self):
        # convenience accessor for env.reset
        return self._env.reset()

    def render(self):
        # convenience accessor for env.render
        self._env.render()

    def observe(self, action):
        # convenience accessor for env.step
        return self._env.step(action)

    @abstractmethod
    def _build(self, config=None):
        NotImplementedError("Abstract.")

    @abstractmethod
    def greedy_policy(self, state):
        NotImplementedError("Abstract.")

    def _buffer_add(self, element):
        if self._buffer_size is None:
            print("Replay buffer size not specified. Defaulting to 1000.")
            self._buffer_size = 1000

        if len(self._replay_buffer) < self._buffer_size:
            self._replay_buffer.append(element)
        else:
            self._replay_buffer[np.random.randint(0, self._buffer_size)] = element

    def _buffer_batch(self, batch_size):
        # if we don't have enough to fill a batch, just return what we've got
        if len(self._replay_buffer) <= batch_size:
            return self._replay_buffer
        return random.sample(self._replay_buffer, batch_size)

    def play(self, n_episodes, max_steps=1000):
        """
        Play some number of episodes with the greedy policy,
        recording and printing the max reward.
        :param n_episodes:
        :param max_steps:
        :return:
        """

        max_reward = 0.0
        for i in range(n_episodes):
            obs, _ = self.reset()
            ep_reward = 0.0
            for j in range(max_steps):
                self.render()
                obs = obs.reshape([1] + self._state_shape)
                action = self.greedy_policy(obs)
                new_obs, reward, done, info, _ = self.observe(action)
                ep_reward += reward
                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    obs = new_obs

        return max_reward

    @property
    def env(self):
        return self._env

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self._buffer_size = value

    @property
    def train_max_reward(self):
        return self._train_max_reward