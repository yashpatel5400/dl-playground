"""
__name__   = predict.py
__author__ = Yash Patel
__description__ = Does the prediction using the defined model and data
"""

import gym
import numpy as np

from data import gather_data
from model import create_model

def predict():
    env = gym.make("CartPole-v0")
    trainingX, trainingY = gather_data(env)
    model = create_model()
    model.fit(trainingX, trainingY, epochs=5)
    
    scores = []
    num_trials = 50
    sim_steps = 500

    for trial in range(num_trials):
        observation = env.reset()
        score = 0
        for step in range(sim_steps):
            action = np.argmax(model.predict(observation.reshape(1,4)))
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
    
    print(np.mean(scores))

if __name__ == "__main__":
    predict()