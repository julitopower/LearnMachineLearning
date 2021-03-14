import argparse as ap
import sys

import gym
import numpy as np

import pong
import pgagent
import actorcriticagent2
import actorcriticagent


class Params(object):
    def __init__(self):
        self.gamma = None
        self.lr = None
        self.lr2 = None
        self.er = None
        self.verbose = False
        self.state_dim = None
        self.actions_dim = None
        self.agent = None


def get_params(args, state_dim, actions_dim):
    params = Params()
    params.gamma = args.gamma
    params.lr = args.lr
    params.lr2 = args.lr2
    params.er = args.er
    params.verbose = args.v
    params.state_dim = state_dim
    params.actions_dim = actions_dim
    params.agent = args.agent
    return params


def get_env(name):
    if name == 'cartpole':
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 500
        return env
    elif name == 'pong':
        return pong.Pong(800, 600, int(400/2), int(200/2))


def get_agent(params):
    if params.agent == 'ac2':
        return actorcriticagent2.DummyAgent(params.state_dim,
                                            params.actions_dim,
                                            params.gamma,
                                            params.lr,
                                            params.lr2,
                                            params.er,
                                            layers_actor=[64, 256, 512, 1024, 2048, 256, 64],
                                            layers_critic=[64, 256, 512, 64])


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Pong game RL")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--lr", help="Learning rate",
                        type=float, required=True)
    parser.add_argument("--lr2", help="Learning rate", type=float)
    parser.add_argument("--er", help="Entropy rate", type=float, required=True)
    parser.add_argument("--env", help="Environment name",
                        type=str, default='cartpole')
    parser.add_argument("--agent", help="Agent name",
                        type=str, default='ac2')
    parser.add_argument("-i", help="Model to load", type=str)
    parser.add_argument("-o", help="Model to save to", type=str)            
    parser.add_argument("-v", help="Visualize pong game",
                        action="store_true", default=False)
    args = parser.parse_args()

    env = get_env(args.env)
    params = get_params(
        args, env.observation_space.shape[0], env.action_space.n)
    agent = get_agent(params)
    try:
        agent.load(args.i)
    except:
        pass
    

    # Setup numpy print options
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    score_history = []
    episodes = 10000
    for i in range(episodes):
        score = 0
        done = False

        obs = env.reset()
        while not done:
            action, probs = agent.action(obs)
            obs_next, reward, done, info = env.step(action)
            agent.record(obs, action, probs, reward)
            obs = obs_next
            score += reward
            if params.verbose:
                env.render()
            if i % 10 == 0:
                try:
                    agent.save(args.o)
                except:
                    pass
                
        score_history.append(score)

        agent.train()
        avg_score = np.mean(score_history[-100:])
        print(f"Episode {i}, score {score}, avg_score {avg_score}")

    print(score_history)
