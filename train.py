
"""
-- Test architectures.
"""

config_ = {
  "env": "CartPole-v0",
  "train_episodes": 500,
  "play_episodes": 10,
  "discount": 0.99,
  "train_kwargs": {
    "epsilon": 0.01,
    "epsilon_schedule": 10
  }
}

agent_tag = "DeepQ"


if __name__ == "__main__":

    import argparse as ap
    import json
    import gym
    import autorl.agents as agents

    parser = ap.ArgumentParser(description='Deep RL!')
    parser.add_argument('--q-learner', dest='agent', default=None, action='store_const',
                        const="DeepQ", help='Flag for q-learning.')
    parser.add_argument('--mc-learner', dest='agent', default=None, action='store_const',
                        const="DeepMC", help='Flag for mc-learning.')
    parser.add_argument('--config', dest='config', action='store', default=None,
                        help='Location of the configuration file.')

    args = parser.parse_args()

    if args.agent is not None:
        agent_tag = args.agent
    else:
        print("No agent flag specified. Defaulting to DeepQ agent.")

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        config_.update(config)

    env = gym.make(config_["env"])

    # instantiate and configure agent, train, and play
    agent = getattr(agents, agent_tag)(env, config_["discount"])
    agent.train(config_["train_episodes"], **config_["train_kwargs"])
    agent.play(config_["play_episodes"])

    print("Finished.")
