"""
deep Q learning implementation in Keras
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

from collections import deque

class DQN:
	def __init__(self, env, gamma, epsilon):
		self.env     = env
		self.gamma   = gamma
		self.epsilon = epsilon
		self.memory  = deque(maxlen=2000)
		self.model   = create_model()
	
	def create_model(self):
		model   = Sequential()
		state_space  = self.env.observation_space.shape
		action_shape = self.env.action_space.shape

		model.add(Dense(128, input_shape=state_space, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(256, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(128, activation="relu"))
		model.add(Dropout(0.8))

		model.add(Dense(action_space, activation="linear"))
		model.compile(loss="mean_squared_error",
			optimizer="adam")
		return model

	def predict(self, state):
		EPSILON_DECAY = .975
		self.epsilon = EPSILON_DECAY * self.epsilon
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return np.argmax(self.model.predict(state))

	def remember(self, state, action, reward, new_state):
		self.memory.append([state, action, reward, new_state])

	def replay(self):
		pass

env     = gym.make("CartPole-v0")
gamma   = 0.9
epsilon = .125

trials  = 1000
trial_len = 500

dqn_agent = DQN(env=env, gamma=gamma, epsilon=epsilon)
for _ in range(trials):
	observation = env.reset()
	for _ in range(trial_len):
		pass

# want to predict valuation of each state/action pair, i.e. minimize: 
# r + y max_{a'} Q(s',a') - Q(s,a)
# y is the gamma decay factor for future valuations

# choose an action
	# use random action w/ prob epsilon
	# otherwise do prediction from the current state w/ argmax value
# after choosing, take action and find new state
# save this into memory as [state, action, reward, next_state]
# train on memory of saved states