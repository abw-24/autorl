
from gym import spaces


class GymAgent(object):

    def __init__(self, env):
        """
        Base RL agent for Gym RL environments. Does some initial introspection
        to set the state and action space info needed for model configuration,
        sets up generic methods and methods to be overwritten by child classes.
        :param env: Gym environment
        """

        self._env = env
        self._state_space = env.observation_space
        self._action_space = env.action_space
        self._value_network = None
        self._policy_network = None

        # check the environment, set shapes and types
        assert isinstance(self._state_space, spaces.Box), \
            "Expecting a Gym `Box` state space."

        self._state_dim = self._state_space.shape

        n_dim = len(self._state_dim)
        if n_dim == 1:
            self._state_type = "vector"
        else:
            self._state_type = "{d}-tensor".format(d=n_dim)

        if isinstance(self._action_space, spaces.discrete.Discrete):
            self._action_dim = self._action_space.n
            self._action_type = "discrete"
        else:
            raise NotImplementedError(
                "Currently only supporting `Discrete` action spaces."
            )

    def reset(self):
        # convenience accessor for env.reset
        self._env.reset()

    def render(self):
        # convenience accessor for env.render
        self._env.render()

    def observe(self, action):
        # convenience accessor for env.step
        return self._env.step(action)

    def _configure(self):
        pass

    def train(self, n_epochs, config):
        pass

    @property
    def env(self):
        return self._env


class MCAgent(GymAgent):

    def __init__(self, env):

        super(MCAgent, self).__init__(env)

