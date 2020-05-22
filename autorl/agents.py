
from gym import spaces
import random
import numpy as np
from nets import nets, train


class GymAgent(object):

    def __init__(self, env, discount):
        """
        Base RL agent for Gym RL environments. Does some initial introspection
        to set the state and action space info needed for model configuration,
        sets up generic methods and methods to be overwritten by child classes.
        :param env: Gym environment
        """

        self._env = env
        self._discount = discount
        self._state_space = env.observation_space
        self._action_space = env.action_space

        # check the environment, set shapes and types
        assert isinstance(self._state_space, spaces.Box), \
            "Expecting a Gym `Box` state space."

        self._state_dim = list(self._state_space.shape)

        n_dim = len(self._state_dim)
        if n_dim == 1:
            self._state_type = "vector"
        else:
            self._state_type = "{d}-tensor".format(d=n_dim)

        if isinstance(self._action_space, spaces.discrete.Discrete):
            self._action_dim = int(self._action_space.n)
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

    def configure(self):
        pass

    def policy(self, state, epsilon):
        pass

    def greedy_policy(self, state):
        pass

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=None, schedule_freq=50):
        pass

    def play(self, n_episodes, max_steps=1000):
        """
        Play some number of episodes with the greedy policy,
        recording and printing the max reward.
        """

        max_reward = 0.0
        for i in range(n_episodes):
            obs = self.reset()
            ep_reward = 0.0
            for i in range(max_steps):
                self.render()
                obs = obs.reshape([1] + self._state_dim)
                action = self.greedy_policy(obs)
                new_obs, reward, done, info = self.observe(action)
                ep_reward += reward
                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    obs = new_obs

    @property
    def env(self):
        return self._env


#################
#################
# VALUE METHODS #
#################
#################


class DeepMC(GymAgent):

    def __init__(self, env, discount=0.99):

        super(DeepMC, self).__init__(env, discount)
        self._q_network = None

    def _return(self, rewards):
        """
        Return the total discounted reward for a given sequential
        list of rewards and a discount rate
        :param rewards: List of rewards
        :param discount: Discount rate
        :return: MC Target (total discounted rewards)
        """
        return sum([rewards[i]*pow(self._discount, i) for i in range(len(rewards))])

    def _batch(self, data):
        """
        Parse the raw state, action, reward episode data into a batch
        for updating the action value network.
        :param data: Data collected from train loop
        :return: Numpy batches
        """

        states, actions, rewards = zip(*data)
        x_array = np.array(states).reshape([len(data)] + self._state_dim)
        y_array = np.zeros((len(data), self._action_dim))

        for i, s in enumerate(states):
            target_vector = self.q_eval(s)
            target_vector[actions[i]] = self._return(rewards[i:])
            y_array[i,:] = target_vector

        return x_array, y_array

    def configure(self, config=None):
        """
        Configure and compile the action value network. For now, a
        simple 2 layer MLP
        :param model_type:
        :return:
        """

        batch_shape = tuple([None] + self._state_dim)

        if self._state_type == "vector":
            if config is None:
                config = {
                        "dense_dims": list(range(self._action_dim+1, self._state_dim[0]+1))[::-1],
                        "dense_activation": "relu",
                        "output_dim": self._action_dim,
                        "optimizer": {"Adam": {"learning_rate": 0.1}},
                        "loss": {"MeanSquaredError": {}},
                }

            network = nets.MLP(config)

        elif "tensor" in self._state_type:
            raise NotImplementedError("Only supporting MLPs right now.")

        else:
            raise ValueError("Unrecognized state type.")

        # compiled model
        self._q_network = train.model_init(network, config, batch_shape)

    def q_eval(self, state):
        """
        Evaluate the q network at a given state
        :param state:
        :return:
        """
        return self._q_network.predict(state).reshape(self._action_dim)

    def greedy_policy(self, state):
        """
        Greedy action for the provided state
        :param state: State
        :return: Action index
        """
        return np.argmax(self.q_eval(state))

    def policy(self, state, epsilon):
        """
        Epsilon-greedy policy
        :param state: State
        :param epsilon: Epsilon (float)
        :return: Action index
        """

        if random.random() < epsilon:
            action = random.randint(0, self._action_dim-1)
        else:
            action = self.greedy_policy(state)

        return action

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=False, schedule_freq=10):
        """
    `   For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. once the episode is up, use the
        true reward to prep a batch and update the action value network.
        :param n_episodes:
        :param max_steps:
        :return:
        """
        if self._q_network is None:
            self.configure()

        max_reward = 0.0
        for i in range(n_episodes):
            if epsilon_schedule and i % schedule_freq == 0:
                schedule_step = int(float(i)/schedule_freq)
                epsilon = epsilon*pow(0.9, schedule_step)
            obs = self.reset()
            ep_reward = 0.0
            data = []
            for j in range(max_steps):
                self.render()
                obs = obs.reshape([1] + self._state_dim)
                action = self.policy(obs, epsilon=epsilon)
                new_obs, reward, done, info = self.observe(action)
                ep_reward += reward
                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    data.append([obs, action, reward])
                    obs = new_obs

            # after the episode, prep batch for training, take the grads, and apply
            batch_x, batch_y = self._batch(data)
            loss, grads = train.grad(self._q_network, batch_x, batch_y)
            updates = zip(grads, self._q_network.trainable_variables)
            self._q_network.optimizer.apply_gradients(updates)

            print("Current loss: {}".format(loss))


class DeepQ(GymAgent):

    def __init__(self, env, discount=0.99):

        super(DeepQ, self).__init__(env, discount)

    def target(self):
        pass


class DeepSARSA(GymAgent):

    def __init__(self, env, discount=0.99):

        super(DeepSARSA, self).__init__(env, discount)

    def target(self):
        pass