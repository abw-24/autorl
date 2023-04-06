# autorl (WIP)

- Simple RL agents for use with Gym, Tensorflow 2.x
- Depends on [`nets`](https://pages.github.com/abw-24/nets) project
- Test suite can be run with `pytest` as the test runner:
```
pytest src/autorl --disable-warnings
```
- Note: Integration tests do not mock training or writing to disk, so the full suite may take a few minutes to run
    
    
### Usage:

```
import gym
from autorl import DeepQ

# Create a gym environment
env = gym.make("CartPole-v1")
# Instantiate an agent (here with the default q-network)
agent = DeepQ(env=env, discount=0.99)
# Train
agent.train(n_episodes=1000, weight_freeze=10)
# Play
agent.play(n_episodes=1)
        
```