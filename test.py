
"""
-- Test architectures on the cartpole problem
"""

import gym
from autorl import agents

N_EPISODES = 500
DISCOUNT = 0.9

env = gym.make('CartPole-v0')

################
# value agents #
################

mc = agents.DeepMC(env=env, discount=DISCOUNT)

mc.train(N_EPISODES, epsilon=0.1, epsilon_schedule=True, schedule_freq=10)
mc.play(100)