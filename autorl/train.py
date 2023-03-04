
"""
Training entry point.
"""

defaults = {
    "env": "CartPole-v0",
    "train_episodes": 500,
    "play_episodes": 10,
    "discount": 0.99,
    "network": None,
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
    import autorl.agents.value as agents

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

    # If a configuration file is specified, read and update
    # the default configuration.
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        defaults.update(config)

    env = gym.make(defaults.get("env"))

    # Instantiate and configure agent, train, and play
    agent = getattr(agents, agent_tag)(
            env=env,
            discount=defaults.get("discount"),
            config=defaults.get("network")
    )
    agent.train(defaults.get("train_episodes"), **defaults.get("train_kwargs"))
    agent.play(defaults.get("play_episodes"))

    print("Finished.")
