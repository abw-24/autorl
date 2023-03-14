
import unittest
import gym

import autorl.agents.value as agents


#TODO: Add mocks for q evaluation and test `batch` method
#TODO: Add tests for DeepQ (`batch` only)
class TestDeepMC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up gym environment once
        cls._env = gym.make("CartPole-v1")

    @classmethod
    def tearDownClass(cls):
        # Set up gym environment once
        del cls._env

    def setUp(self):
        self._discount = 0.5
        self._default_network_config = None

        self._default_agent = agents.DeepMC(
                env=self._env,
                discount=self._discount,
                config=self._default_network_config
        )

    def test_agent_return(self):
        input_cases = [[0,0,1], [1,0,0]]
        output_cases = [0.25, 1.0]
        for i, o in zip(input_cases, output_cases):
            reward = self._default_agent._return(i)
            assert reward == o, "Discounted return value mismatch. " \
                                "Expected {}, got {}.".format(o, reward)

