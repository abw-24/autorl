
from gym import spaces
import random
import numpy as np
from nets import nets, train
from tensorflow.keras.models import clone_model


class GymAgent(object):

    def __init__(self, env, discount=0.99):
        """
        Base RL agent for Gym RL environments. Does some initial introspection
        to set the state and action space info needed for model configuration,
        sets up generic methods and methods to be overwritten by child classes.
        :param env: Gym environment
        :param discount: Reward discounts
        """

        self._env = env
        self._discount = discount
        self._state_space = env.observation_space
        self._action_space = env.action_space
        self._replay_buffer = []
        self._buffer_size = None

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

    def greedy_policy(self, state):
        pass

    def _buffer_add(self, triplet):
        if len(self._replay_buffer) < self._buffer_size:
            self._replay_buffer.append(triplet)
        else:
            if self._buffer_size is None:
                print("Replay buffer size not specified. Defaulting to 1000.")
                self._buffer_size = 1000
            self._replay_buffer[np.random.randint(0, self._buffer_size)] = triplet

    def _buffer_batch(self, batch_size):
        if len(self._replay_buffer) < batch_size:
            return self._replay_buffer
        return random.sample(self._replay_buffer, batch_size)

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

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def buffer_size(self):
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        self._buffer_size = value


#################
#################
# VALUE METHODS #
#################
#################


class ValueAgent(GymAgent):

    def __init__(self, env, discount=0.99):
        super(ValueAgent, self).__init__(env, discount)
        self._q_network = None
        self._frozen_q_network = None

    def configure(self, config=None):
        """
        Configure and compile the action value network. For now, just an
        MLP, but CNNs will be supported soon.
        :param model_type:
        :return:
        """

        batch_shape = tuple([None] + self._state_dim)

        if self._state_type == "vector":
            config_ = {
                        "dense_dims": list(range(self._action_dim+1, self._state_dim[0]+1))[::-1],
                        "dense_activation": "relu",
                        "output_dim": self._action_dim,
                        "optimizer": {"Adam": {"learning_rate": 0.01}},
                        "loss": {"MeanSquaredError": {}},
            }

            if config is not None:
                config_.update(config)

            network = nets.MLP(config_)

        elif "tensor" in self._state_type:
            raise NotImplementedError("Only supporting MLPs right now.")

        else:
            raise ValueError("Unrecognized state type.")

        # compiled model
        self._q_network = train.model_init(network, config_, batch_shape)

    def q_eval(self, state, frozen=False):
        """
        Evaluate the q network (or frozen q network) at the current state
        :param state:
        :return:
        """
        if frozen:
            assert self._frozen_q_network is not None, \
                "Asked for the frozen q network evaluation, but no frozen q network available."

            return self._frozen_q_network.predict(state).reshape(self._action_dim)
        else:
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


class DeepMC(ValueAgent):

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

        # iterate over states and construct mc-target
        for i, s in enumerate(states):
            target_vector = self.q_eval(s)
            target_vector[actions[i]] = self._return(rewards[i:])
            y_array[i,:] = target_vector

        return x_array, y_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=False):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. once the episode is up, use the
        true reward to prep a batch and update the action value network.
        :param n_episodes:
        :param max_steps:
        :param epsilon:
        :param epsilon_schedule:
        :param schedule_freq:
        :return:
        """

        if self._q_network is None:
            self.configure()

        max_reward = 0.0

        for i in range(n_episodes):

            if epsilon_schedule is not None:
                if i % epsilon_schedule == 0:
                    schedule_step = int(i/epsilon_schedule)
                    epsilon = epsilon*pow(0.9, schedule_step)

            obs = self.reset()
            ep_reward = 0.0
            tuple_batch = []

            for j in range(max_steps):
                self.render()
                obs = obs.reshape([1] + self._state_dim)
                action = self.policy(obs, epsilon=epsilon)
                new_obs, reward, done, info = self.observe(action)
                ep_reward += reward
                tuple_batch.append((obs, action, reward))
                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    obs = new_obs

            # after the episode, prep batch for training, take the grads, and apply.
            # because the state space is continuous, essentially "first-visit" MC
            # is all we can really do without bucketing or adding a vector
            # similarity calculation to the processing. possible for future versions
            batch_x, batch_y = self._batch(tuple_batch)
            loss, grads = train.grad(self._q_network, batch_x, batch_y)
            updates = zip(grads, self._q_network.trainable_variables)
            self._q_network.optimizer.apply_gradients(updates)

            print("Current loss: {}".format(loss))


class DeepQ(ValueAgent):
    """
    Deep Q-Learning with experience replay and optional weight freezing.
    """

    def __init__(self, env, discount=0.99):
        super(DeepQ, self).__init__(env, discount)

    def _batch(self, data, q_freeze):
        """
        Construct a batch for learning using the provided tuples and
        the action value function.
        :param data:
        :param q_freeze:
        :return:
        """

        states, actions, rewards, states_prime = zip(*data)
        x_array = np.array(states).reshape([len(data)] + self._state_dim)
        y_array = np.zeros((len(data), self._action_dim))

        # iterate over states and construct q learning target
        for i, s in enumerate(states):
            target_vector = self.q_eval(s, q_freeze is not None)
            target_prime = self.q_eval(states_prime[i], q_freeze is not None)
            target_vector[actions[i]] = rewards[i] + self._discount*np.max(target_prime)
            y_array[i,:] = target_vector

        return x_array, y_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=10, buffer_size=128,
              batch_size=16, q_freeze=None):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. At each step, use the action value
        function, the observed tuple, and a set of random tuples from the
        replay buffer (experience replay) to bootstrap the q-learning targets
        and update the q network. Target network can be optionally frozen for
        a specified number of steps.

        :param n_episodes:
        :param max_steps:
        :param epsilon:
        :param epsilon_schedule:
        :param buffer_size:
        :param batch_size:
        :param q_freeze:
        :return:
        """
        if self._q_network is None:
            self.configure()

        self._buffer_size = buffer_size

        if q_freeze is not None:
            self._frozen_q_network = clone_model(self._q_network)
            self._frozen_q_network.set_weights(self._q_network.weights)

        max_reward = 0.0
        total_steps = 0
        loss = 0.0

        for i in range(n_episodes):

            if epsilon_schedule is not None:
                if i % epsilon_schedule == 0:
                    schedule_step = int(float(i)/epsilon_schedule)
                    epsilon = epsilon*pow(0.9, schedule_step)

            obs = self.reset()
            obs = obs.reshape([1] + self._state_dim)
            ep_reward = 0.0

            for j in range(max_steps):
                total_steps += 1

                self.render()
                action = self.policy(obs, epsilon=epsilon)
                new_obs, reward, done, info = self.observe(action)
                new_obs = new_obs.reshape([1] + self._state_dim)
                ep_reward += reward

                # with experience replay, we sample (batch_size - 1) records to
                # pair with the new observation. afterwards, add to the
                # replay buffer (evicts an old record if already full)
                current_tuple = (obs, action, reward, new_obs)
                tuple_batch = self._buffer_batch(batch_size-1)
                tuple_batch.append(current_tuple)
                self._buffer_add(current_tuple)

                batch_x, batch_y = self._batch(tuple_batch, q_freeze)
                loss, grads = train.grad(self._q_network, batch_x, batch_y)
                updates = zip(grads, self._q_network.trainable_variables)
                self._q_network.optimizer.apply_gradients(updates)

                # if we're using frozen weights, check if its time to update
                if q_freeze is not None:
                    if total_steps % q_freeze == 0:
                        self._frozen_q_network.set_weights(self._q_network.weights)

                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    obs = new_obs

            print("Finished game {}...".format(i+1))
            print("Current loss: {}".format(loss))
