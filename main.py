import gymnasium as gym
from dqn import Agent
from gymnasium.envs import box2d
#from utils import plot_learning_curve

import numpy as np

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4,
     eps_min=0.03, input_dims=[8], lr=0.001)

    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation, info = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            if truncated:
                observation_, info = env.reset()
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print("episode ", i, "score %.2f" % score, "average_score %.2f" % avg_score,
         "epsilon %.2f" % agent.epsilon)


    env.close()