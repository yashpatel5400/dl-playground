"""
deep Q learning implementation in Keras
"""

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout

from collections import deque

class DQN:
	def __init__(self, env, gamma, epsilon):
		self.env     = env
		self.gamma   = gamma
		self.epsilon = epsilon
		self.memory  = deque(maxlen=2000)
		self.model   = self.create_model()
	
	def create_model(self):
		model   = Sequential()
		state_shape  = self.env.observation_space.shape

		model.add(Dense(24, input_shape=state_shape, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(48, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(24, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(self.env.action_space.n))
		model.compile(loss="mean_squared_error",
			optimizer="adam")
		return model

	def predict(self, state):
		EPSILON_DECAY = .99
		self.epsilon = EPSILON_DECAY * self.epsilon
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return np.argmax(self.model.predict(state)[0])

	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	def retrain(self):
		batch_size = 32
		if len(self.memory) < batch_size:
			return

		samples = random.sample(self.memory, batch_size)

		states  = []
		targets = []
		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.model.predict(state)
			if done: # no future to predict in this case
				target[0][action] = reward
			else:
				Q_future = max(self.model.predict(new_state)[0])
				target[0][action] = reward + Q_future * gamma
		
			states.append(state)
			targets.append(target)

		for state, target in zip(states, targets):
			self.model.fit(state, target, epochs=1, verbose=0)

env     = gym.make("CartPole-v0")
gamma   = 0.9
epsilon = .95

trials  = 1000
trial_len = 500

min_steps = 199
dqn_agent = DQN(env=env, gamma=gamma, epsilon=epsilon)
for trial in range(trials):
	cur_state = env.reset().reshape(1,4)
	for step in range(trial_len):
		action = dqn_agent.predict(cur_state)
		new_state, reward, done, _ = env.step(action)

		new_state = new_state.reshape(1,4)		
		dqn_agent.remember(cur_state, action, reward, new_state, done)
		dqn_agent.retrain()
		cur_state = new_state
		if done:
			break
	if step == min_steps:
		break
	print("Successfully ran {} steps".format(step))

print("Completed in {} trials".format(trial))