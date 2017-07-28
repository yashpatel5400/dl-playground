"""
__name__   = data.py
__author__ = Yash Patel
__description__ = Gathers the data for the Cartpole environment into the 
X and Y numpy arrays for training
"""

import gym
import numpy as np

def gather_data(env):
    min_score = 50
    sim_steps = 500
    trainingX, trainingY = [], []

    scores = []
    for _ in range(10000):
        observation = env.reset()
        score = 0
        training_sampleX, training_sampleY = [], []
        for step in range(sim_steps):
            # action corresponds to the previous observation so record before step
            action = np.random.randint(0, 2)
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1
            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)
            
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > min_score:
            scores.append(score)
            trainingX += training_sampleX
            trainingY += training_sampleY

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    return trainingX, trainingY

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    trainingX, trainingY = gather_data(env)