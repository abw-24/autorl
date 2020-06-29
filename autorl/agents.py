
import numpy as np
import random
from nets import nets, train
from autorl.base import GymAgent


#################
# VALUE METHODS #
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
        :return:
        """

        batch_shape = tuple([None] + self._state_dim)

        if self._state_type == "vector":
            config_ = {
                        "dense_dims": list(range(self._action_dim+1, self._state_dim[0]+1))[::-1],
                        "dense_activation": "relu",
                        "output_dim": self._action_dim,
                        "optimizer": {"Adam": {"learning_rate": 0.001}},
                        "loss": {"MeanSquaredError": {}},
                        "output_activation": "linear"
            }

            if config is not None:
                config_.update(config)

            network = nets.MLP(config_)

        elif "tensor" in self._state_type:
            raise NotImplementedError("Only supporting MLPs right now.")

        else:
            raise ValueError("Unrecognized state type.")

        # compiled model
        return train.model_init(network, config_, batch_shape)

    def _set_frozen_weights(self):
        self._frozen_q_network.set_weights(
                [w.numpy() for w in self._q_network.weights]
        )

    def q_eval(self, state, freeze_flag=False, reshape=None):
        """
        Evaluate the q network (or frozen q network) at the current state array
        :param state:
        :return:
        """
        if freeze_flag:
            assert self._frozen_q_network is not None, \
                "Asked for the frozen q network evaluation, but no frozen q network available."
            preds = self._frozen_q_network.predict(state)

        else:
            preds = self._q_network.predict(state)

        if reshape is not None:
            preds = preds.reshape(reshape)

        return preds

    def greedy_policy(self, state):
        """
        Greedy action for the provided state
        :param state: State
        :return: Action index
        """
        return np.argmax(self.q_eval(state, reshape=self._action_dim))

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

    def _return(self, rewards):
        """
        Return the total discounted reward for a given sequential
        list of rewards and a discount rate
        :param rewards: List of rewards
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
        state_array = np.array(states).reshape([len(data)] + self._state_dim)
        target_array = np.zeros((len(data), self._action_dim))

        # iterate over states and construct mc-target
        for i, s in enumerate(states):
            target_vector = self.q_eval(s, reshape=self._action_dim)
            target_vector[actions[i]] = self._return(rewards[i:])
            target_array[i,:] = target_vector

        return state_array, target_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=False, network=None):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. Once the episode is up, use the
        true reward to prep a batch and update the action value network.
        :param n_episodes:
        :param max_steps:
        :param epsilon:
        :param epsilon_schedule:
        :return:
        """

        if self._q_network is None:
            self._q_network = self.configure(network)

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
        state_shape = [len(data)] + self._state_dim
        action_shape = [len(data), self._action_dim]

        # get the current q values for the current state and next
        # state over the entire batch simultaneously
        state_array = np.array(states).reshape(state_shape)
        state_prime_array = np.array(states_prime).reshape(state_shape)
        target_array = self.q_eval(state_array, freeze_flag=q_freeze is not None,
                                   reshape=action_shape)
        target_prime_array = self.q_eval(state_prime_array, freeze_flag=q_freeze is not None,
                                         reshape=action_shape)

        # iterate and update target array with rewards plus
        # the discounted max over the next state's q values
        for i in range(target_prime_array.shape[0]):
            # place the rewards in the index associated with the action taken.
            # the rest of the targets are the existing q values (so they aren't involved
            # when taking the gradient wrt the loss and only the action taken gets updated)
            target_array[i, actions[i]] = rewards[i] + self._discount*np.max(target_prime_array[i,:])

        return state_array, target_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=10, buffer_size=128,
              batch_size=16, weight_freeze=None, network=None):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. At each step, use the action value
        function and a set of random 4-tuples from the replay buffer to bootstrap
        the q-learning targets and update the network.

        :param n_episodes:
        :param max_steps:
        :param epsilon:
        :param epsilon_schedule:
        :param buffer_size:
        :param batch_size:
        :param weight_freeze:
        :param network:
        :return:
        """
        if self._q_network is None:
            self._q_network = self.configure(network)

        self._buffer_size = buffer_size

        if weight_freeze is not None:
            self._frozen_q_network = self.configure(network)
            self._set_frozen_weights()

        max_reward = 0.0
        total_steps = 0
        freeze_step = 0
        loss = 0.0

        for i in range(n_episodes):

            if epsilon_schedule is not None:
                if i % epsilon_schedule == 0:
                    epsilon = epsilon*0.999

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

                # add to the replay buffer (evicts an old record if already full)
                # and sample to generate a batch for building targets and training
                self._buffer_add((obs, action, reward, new_obs))
                tuple_batch = self._buffer_batch(batch_size)
                batch_x, batch_y = self._batch(tuple_batch, weight_freeze)

                # train
                loss, grads = train.grad(self._q_network, batch_x, batch_y)
                updates = zip(grads, self._q_network.trainable_variables)
                self._q_network.optimizer.apply_gradients(updates)

                if done:
                    max_reward = max(max_reward, ep_reward)
                    print("Current max reward: {}".format(max_reward))
                    break
                else:
                    obs = new_obs

            print("Finished game {}...".format(i+1))
            print("Current loss: {}".format(loss))

            # if we're using frozen weights, check if its time to update
            # by dividing the total steps by the weight freeze param
            # and taking the floor. if the integer result is greater
            # than the current freeze_step integer value, then it's time
            if weight_freeze is not None:
                current_step = int(total_steps / weight_freeze)
                if current_step > freeze_step:
                    self._set_frozen_weights()
                    freeze_step = current_step
