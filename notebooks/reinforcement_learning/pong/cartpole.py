import gym
import numpy as np
import pgagent
import sys

if __name__ == "__main__":
    gamma = float(sys.argv[1])
    lr = float(sys.argv[2])
    entropy_c = float(sys.argv[3])
  
    env = gym.make('CartPole-v0')
    agent = pgagent.DummyAgent(
        env.observation_space.shape[0], env.action_space.n, gamma, lr, entropy_c)
    score_history = []
    episodes = 2000
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
        score_history.append(score)

        agent.train()
        avg_score = np.mean(score_history[-100:])
        print(f"Episode {i}, score {score}, avg_score {avg_score}")

    print(score_history)
