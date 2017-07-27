"""
a first attempt at
solving the pendulum also with Deep Q Learning (without reference this time)
"""

import gym
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from collections import deque

class DQN:
	def __init__(self, env):
		self.env = env

		self.learning_rate = 0.001
		self.gamma = 0.90
		self.epsilon = .95
		self.epsilon_decay = .99

		self.model = self.create_model()
		self.memory = deque(maxlen=2000)
		self.batch_size = 32

	def create_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.env.observation_space.shape[0], 
			activation="relu"))
		model.add(Dense(48, activation="relu"))
		model.add(Dense(24, activation="relu"))
		model.add(Dense(self.env.action_space.shape[0]))
		model.compile(loss="mean_squared_error",
			optimizer=Adam(lr=self.learning_rate))
		return model

	def predict(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return self.model.predict(cur_state)

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def learn(self):
		if len(self.memory) < self.batch_size:
			return

		samples = random.sample(self.memory, self.batch_size)
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			target = self.model.predict(cur_state)
			print(action)
			if done:
				target[0][action] = reward
			else:
				print(target[0])
				target[0][action] = reward \
					+ self.gamma * np.max(self.model.predict(new_state)[0])
			self.model.train(cur_state, target)

num_trials = 10000
trial_len  = 500
env_name   = "Pendulum-v0"

env = gym.make(env_name)
dqn_agent = DQN(env)

for trial in range(num_trials):
	cur_state = env.reset()
	for step in range(trial_len):
		cur_state = cur_state.reshape(1, env.observation_space.shape[0])
		action = dqn_agent.predict(cur_state)
		
		new_state, reward, done, _ = env.step(action)
		new_state = new_state.reshape(1, env.observation_space.shape[0])

		dqn_agent.remember(cur_state, action, reward, new_state, done)
		dqn_agent.learn()
		
		cur_state = new_state
		if done:
			break

	print("Completed {} steps".format(step))