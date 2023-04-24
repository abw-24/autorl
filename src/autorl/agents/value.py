import random

import numpy as np
import tensorflow as tf
from nets.models.mlp import MLP
from nets.tests.utils import obj_from_config

from autorl.agents.base import GymAgentABC, GymAgent


class ValueAgent(GymAgent, GymAgentABC):
    """
    Base class for action-value-function control algorithms (methods with
    no explicit policy representation).
    """

    def __init__(self, env, discount=0.99, config=None):
        super().__init__(env, discount, config)

        self._q_network = self._build(self._config)
        self._frozen_q_network = None
        self._freeze_flag = False

    def _set_frozen_weights(self):
        """
        Transfer current q network weights to frozen network.
        :return: NoReturnValue
        """
        self._frozen_q_network.set_weights(
                [w.numpy() for w in self._q_network.weights]
        )

    def _build(self, config=None):
        """
        Configure and compile the action value network.
        :return: Compiled TF Keras model
        """

        input_shape = tuple([None] + self._state_shape)

        if self._state_type == "vector":
            # Simple defaults for hidden dimensions if not provided (for now)
            hidden_dims = sorted(
                    [self._action_shape * 2, int(self._state_shape[0] / 2)],
                    reverse=True
            )
            defaults = {
                "hidden_dims": hidden_dims,
                "activation": "relu",
                "output_dim": self._action_shape,
                "optimizer": {"Adam": {"learning_rate": 0.001}},
                "loss": {"MeanSquaredError": {}},
                "output_activation": "relu"
            }

            # If provided a config, use to update the default values
            if config is not None:
                defaults.update(config)

            # Pass default config, excluding compilation key/vals
            exclude = ["optimizer", "loss"]
            network = MLP.from_config({
                k: v for k, v in defaults.items() if k not in exclude
            })
            network.build(input_shape)
            network.compile(
                    loss=obj_from_config(
                            tf.keras.losses, defaults.get("loss")
                    ),
                    optimizer=obj_from_config(
                            tf.keras.optimizers, defaults.get("optimizer")
                    )
            )

        elif "tensor" in self._state_type:
            raise NotImplementedError("Only supporting vector inputs currently.")

        else:
            raise ValueError("Unrecognized state type.")

        return network

    def q_eval(self, state, reshape=None):
        """
        Evaluate the q network (or frozen q network) at the current state array
        :param state: State array (or array of state arrays)
        :param freeze_flag: Boolean for whether we should use a frozen network for eval
        :param reshape: A shape to reshape the output to (if needed)
        :return: Action value predictions
        """
        if self._freeze_flag:
            assert self._frozen_q_network is not None, \
                "Asked for the frozen q network evaluation, " \
                "but no frozen q network available."
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
        return np.argmax(self.q_eval(state, reshape=self._action_shape))

    def policy(self, state, epsilon):
        """
        Epsilon-greedy policy
        :param state: State
        :param epsilon: Epsilon (float)
        :return: Action index
        """

        if random.random() < epsilon:
            action = random.randint(0, self._action_shape - 1)
        else:
            action = self.greedy_policy(state)

        return action

    @property
    def q_network(self):
        return self._q_network


class DeepMC(ValueAgent):

    def __init__(self, env, discount=0.99, config=None):

        super().__init__(env, discount, config)

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
        state_array = np.array(states).reshape([len(data)] + self._state_shape)
        target_array = np.zeros((len(data), self._action_shape))

        # Iterate over states and construct mc-target
        #TODO: refactor to compute targets over whole state trajectory simultaneously
        for i, s in enumerate(states):
            # Place the rewards in the index associated with the action taken.
            # The rest of the targets are the existing q values (so they aren't
            # involved when taking the gradient wrt the loss and only the
            # action taken gets updated)
            target_vector = self.q_eval(s, reshape=self._action_shape)
            target_vector[actions[i]] = self._return(rewards[i:])
            target_array[i,:] = target_vector

        return state_array, target_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=None):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. Once the episode is up, use the
        true reward to prep a batch and update the action value network.
        """
        # Reset train max reward to 0.0
        self._train_max_reward = 0.0

        for i in range(n_episodes):
            if epsilon_schedule is not None:
                if i % epsilon_schedule == 0:
                    schedule_step = int(i/epsilon_schedule)
                    epsilon = epsilon*pow(0.9, schedule_step)

            obs, _ = self.reset()
            ep_reward = 0.0
            tuple_batch = []

            for j in range(max_steps):
                self.render()
                obs = obs.reshape([1] + self._state_shape)
                action = self.policy(obs, epsilon=epsilon)
                new_obs, reward, done, info, _ = self.observe(action)
                ep_reward += reward
                tuple_batch.append((obs, action, reward))
                if done:
                    self._train_max_reward = max(self._train_max_reward, ep_reward)
                    print("Current max reward: {}".format(self._train_max_reward))
                    break
                else:
                    obs = new_obs

            # After the episode, prep batch for training, take the grads, and apply.
            # Because the state space is continuous, essentially "first-visit" MC
            # is all we can really do without bucketing or adding a vector
            # similarity calculation to the processing.
            batch_x, batch_y = self._batch(tuple_batch)
            self._q_network.train_on_batch(batch_x, batch_y)

            print("Finished game {}...".format(i+1))
            print("Current metrics: {}".format(
                    self._q_network.get_metrics_result()
            ))


class DeepQ(ValueAgent):
    """
    Deep Q-Learning with experience replay and optional weight freezing.
    """

    def __init__(self, env, discount=0.99, config=None):

        super(DeepQ, self).__init__(env, discount, config)

    def _batch(self, data):
        """
        Construct a batch for learning using the provided tuples and
        the action value function.
        :param data:
        :param q_freeze:
        :return:
        """

        states, actions, rewards, states_prime = zip(*data)
        state_shape = [len(data)] + self._state_shape
        action_shape = [len(data), self._action_shape]

        # Get the current q values for the current state and next
        # state over the entire batch simultaneously
        state_array = np.array(states).reshape(state_shape)
        state_prime_array = np.array(states_prime).reshape(state_shape)
        target_array = self.q_eval(state_array,reshape=action_shape)
        target_prime_array = self.q_eval(state_prime_array,reshape=action_shape)

        # Iterate and update target array with rewards plus
        # the discounted max over the next state's q values
        for i in range(target_prime_array.shape[0]):
            # Place the rewards in the index associated with the action taken.
            # The rest of the targets are the existing q values (so they aren't
            # involved when taking the gradient wrt the loss and only the
            # action taken gets updated)
            target_array[i, actions[i]] = \
                rewards[i] + self._discount*np.max(target_prime_array[i,:])

        return state_array, target_array

    def train(self, n_episodes, max_steps=1000, epsilon=0.01, epsilon_schedule=10,
              buffer_size=128, batch_size=16, weight_freeze=5000):
        """
        For each episode, play with the epsilon-greedy policy and record
        the states, actions, and rewards. At each step, use the action value
        function and a set of random 4-tuples from the replay buffer to bootstrap
        the q-learning targets and update the network.
        """

        self._buffer_size = buffer_size
        self._freeze_flag = weight_freeze is not None
        self._batch_size = batch_size

        if self._freeze_flag:
            self._frozen_q_network = self._build(self._q_network.get_config())
            self._set_frozen_weights()

        # Reset train max reward to 0.0
        self._train_max_reward = 0.0
        total_steps = 0
        current_swap_step = 0

        for i in range(n_episodes):

            if epsilon_schedule is not None:
                if i % epsilon_schedule == 0:
                    epsilon = epsilon*0.999

            obs, _ = self.reset()
            obs = obs.reshape([1] + self._state_shape)
            ep_reward = 0.0

            for j in range(max_steps):
                total_steps += 1

                self.render()
                action = self.policy(obs, epsilon=epsilon)
                new_obs, reward, done, info, _ = self.observe(action)
                new_obs = new_obs.reshape([1] + self._state_shape)
                ep_reward += reward

                # Add to the replay buffer (evicts an old record if already full)
                # and sample to generate a batch for building targets and training
                self._buffer_add((obs, action, reward, new_obs))
                tuple_batch = self._buffer_batch(self._batch_size)
                batch_x, batch_y = self._batch(tuple_batch)

                # train
                self._q_network.train_on_batch(batch_x, batch_y)

                if done:
                    self._train_max_reward = max(self._train_max_reward, ep_reward)
                    break
                else:
                    obs = new_obs

            print("Finished game {}...".format(i+1))
            print("Current q-network metrics: {}".format(
                    self._q_network.get_metrics_result()
            ))

            # If we're using frozen weights, check if its time to update
            # by dividing the total steps by the weight freeze param
            # and taking the floor. if the integer result is greater
            # than the current freeze_step integer value, then it's time
            if self._freeze_flag:
                episode_swap_step = int(total_steps / weight_freeze)
                if episode_swap_step > current_swap_step:
                    self._set_frozen_weights()
                    current_swap_step += 1

    @property
    def frozen_q_network(self):
        return self._frozen_q_network
