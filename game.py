"""
basic game for learning reinforcement learning
"""

import numpy as np
#from keras.models import Sequential
#from keras.layers import Dense, Convolution
import gym

env = gym.make("CartPole-v0")
best_params = [0 for _ in range(4)]
max_steps = 0

# basic

for _ in range(1000):
	observation = env.reset()
	params = np.random.random(4)
	for step in range(1000): 
		action = int(np.dot(params, observation) > 0)
		observation, reward, done, info = env.step(action)
		if step > max_steps:
			max_steps = step
			best_params = params
		if done:
			break
	if not done:
		break

params = np.random.random(4)
observation = env.reset()
for _ in range(1000):
	action = int(np.dot(best_params, observation) > 0)
	observation, _, _, _ = env.step(action)
	env.render()

print("Completed {} steps".format(steps))