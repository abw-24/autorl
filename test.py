
"""
-- Test architectures on the cartpole problem
--
"""

import gym
import argparse as ap

from nn import conv, nn

N_EPISODES = 5
N_STEPS = 1000

# supported_architectures = ("a2c", "dqn", "random")

###################
# launch cartpole #
###################

env = gym.make('CartPole-v0')

for e in xrange(N_EPISODES):

    env.reset()

    for _ in xrange(N_STEPS):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode {} finished after {} timesteps".format(e, _+1)
            break

    print "Episode {} reached the max allowed steps".format(e)