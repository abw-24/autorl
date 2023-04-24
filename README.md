# autorl (WIP)

- Simple value-based RL agents for use with Gym, Tensorflow 2.x
- Environment introspection allows for easy setup
    - Ultimate goal is to facilitate a bandit learning to train agents
- Depends on [`nets`](https://pages.github.com/abw-24/nets) project
- Test suite can be run with `pytest` as the test runner:
```
pytest src/autorl --disable-warnings
```
- Note: Integration tests do not mock training or saving, so the full suite may take a few minutes to run
    
    
### Example Usage:

```python
import gym
from autorl import DeepQ

# Create a gym environment
env = gym.make("CartPole-v1")
# Instantiate an agent (here with the default q-network)
agent = DeepQ(env=env, discount=0.99)
# Train a frozen Q agent, freezing weights 100 steps at a time
agent.train(n_episodes=1000, weight_freeze=100)
# Play
agent.play(n_episodes=1)

# Save the q-network for other purposes (e.g. serving)
agent.q_network.save("./q-network/")
        
```