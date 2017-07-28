"""
deep Q learning implementation in Keras
"""

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

class DQN:
	def __init__(self, env):
		self.env     = env
		self.memory  = deque(maxlen=2000)
		
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001

		self.model   = self.create_model()

	def create_model(self):
		model   = Sequential()
		state_shape  = self.env.observation_space.shape
		model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
		model.add(Dense(48, activation="relu"))
		model.add(Dense(24, activation="relu"))
		model.add(Dense(self.env.action_space.n))
		model.compile(loss="mean_squared_error",
			optimizer=Adam(lr=self.learning_rate))
		return model

	def act(self, state):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)

		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return np.argmax(self.model.predict(state)[0])

	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	def replay(self):
		batch_size = 32
		if len(self.memory) < batch_size:
			return

		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.model.predict(state)
			if done: # no future to predict in this case
				target[0][action] = reward
			else:
				Q_future = max(self.model.predict(new_state)[0])
				target[0][action] = reward + Q_future * gamma
			self.model.fit(state, target, epochs=1, verbose=0)

env     = gym.make("CartPole-v0")
gamma   = 0.9
epsilon = .95

trials  = 1000
trial_len = 500

min_steps = 199
dqn_agent = DQN(env=env)
for trial in range(trials):
	cur_state = env.reset().reshape(1,4)
	for step in range(trial_len):
		action = dqn_agent.act(cur_state)
		new_state, reward, done, _ = env.step(action)

		new_state = new_state.reshape(1,4)		
		dqn_agent.remember(cur_state, action, reward, new_state, done)
		dqn_agent.replay()
		cur_state = new_state
		if done:
			break
	if step == min_steps:
		break
	print("Successfully ran {} steps".format(step))
print("Completed in {} trials".format(trial))