
"""
-- Test architectures on the cartpole problem
"""

import gym

N_EPISODES = 5
N_STEPS = 1000

###################
# launch cartpole #
###################

env = gym.make('CartPole-v0')

for e in range(N_EPISODES):

    env.reset()

    for _ in range(N_STEPS):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode {} finished after {} timesteps".format(e, _+1))
            break
    else:
        print("Episode {} reached the max allowed steps".format(e))