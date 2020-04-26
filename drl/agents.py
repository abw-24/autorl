"""

"""

class BaseAgent(object):
    """

    """

    def __init__(self, sess, env):

        self._env = env
        self._sess = sess

    def _introspection(self):

        pass

    def observe(self, action):

        # simple accessor for env.step
        return self._env.step(action)

    def _network_eval(self, network, in_array, shape=None):

        pass


class MCAgent(BaseAgent):

    def __init__(self, sess, env):

        super(MCAgent, self).__init__(sess, env)

