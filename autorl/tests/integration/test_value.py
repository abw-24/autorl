
import gym
import unittest

from autorl.agents.value import DeepMC, DeepQ
from autorl.tests.utils import try_except_assertion_decorator


class TestDeepMC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load gym environment once for all DeepMC tests
        """
        cls._env = gym.make("CartPole-v1")

    @classmethod
    def tearDownClass(cls):
        """
        Delete gym environment
        """
        del cls._env

    def setUp(self):
        self._discount = 0.99
        self._default_network_config = None

    def _generate_default_agent(self):
        return DeepMC(
                env=self._env,
                discount=self._discount,
                config=self._default_network_config
        )

    @try_except_assertion_decorator
    def test_build_basic_with_constructor_defaults(self):
        _ = DeepMC(self._env)

    @try_except_assertion_decorator
    def test_build_basic(self):
        _ = self._generate_default_agent()

    def test_build_complex(self):
        test_field = "hidden_dims"
        test_value = [1]
        agent = DeepMC(
                env=self._env,
                discount=self._discount,
                config={test_field: test_value}
        )
        model_config = agent.q_network.get_config()

        assert list(model_config.get(test_field)) == test_value, \
            "Network configuration defaults not being overwritten by " \
            "constructor configuration"

    @try_except_assertion_decorator
    def test_train_and_play_basic(self):
        agent = self._generate_default_agent()
        agent.train(n_episodes=2)
        _ = agent.play(n_episodes=1)

    @try_except_assertion_decorator
    def test_train_and_play_complex(self):
        agent = self._generate_default_agent()
        agent.train(n_episodes=2, epsilon=0.1, epsilon_schedule=1)
        _ = agent.play(n_episodes=1)


class TestDeepQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load gym environment once for all DeepMC tests
        """
        cls._env = gym.make("CartPole-v1")

    @classmethod
    def tearDownClass(cls):
        """
        Delete gym environment
        """
        del cls._env

    def setUp(self):
        self._discount = 0.99
        self._default_network_config = None

    def _generate_default_agent(self):
        return DeepQ(
                env=self._env,
                discount=self._discount,
                config=self._default_network_config
        )

    @try_except_assertion_decorator
    def test_build_basic_with_constructor_defaults(self):
        _ = DeepQ(self._env)

    @try_except_assertion_decorator
    def test_build_basic(self):
        _ = self._generate_default_agent()

    def test_build_complex(self):
        test_field = "hidden_dims"
        test_value = [1]
        agent = DeepQ(
                env=self._env,
                discount=self._discount,
                config={test_field: test_value}
        )
        model_config = agent.q_network.get_config()

        assert list(model_config.get(test_field)) == test_value, \
            "Network configuration defaults not being overwritten by " \
            "constructor configuration"

    @try_except_assertion_decorator
    def test_train_and_play_basic(self):
        agent = self._generate_default_agent()
        agent.train(n_episodes=2)
        _ = agent.play(n_episodes=1)

    def test_train_and_play_complex(self):
        agent = self._generate_default_agent()
        agent.train(
                n_episodes=2,
                weight_freeze=1,
                epsilon_schedule=1,
                batch_size=32,
                buffer_size=256
        )
        assert agent.buffer_size == 256, "Train param `buffer_size` does not" \
                                         "correctly overwrite instance var."
        assert agent.batch_size == 32, "Train param `batch_size` does not" \
                                         "correctly overwrite instance value."
        assert agent._freeze_flag, "Internal freeze_flag should be set if train" \
                                   " param `weight_freeze is not None"
        assert agent.frozen_q_network is not None, "Frozen q-network was not set."
        assert agent.frozen_q_network.get_config() == agent.q_network.get_config(), \
            "Q-network and frozen q-network do not have equivalent configurations."

        _ = agent.play(n_episodes=1)