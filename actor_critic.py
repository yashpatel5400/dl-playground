"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add
from keras.optimizers import Adam

# given the current state, determines the best action to take 
def create_actor_model(env):
	model = Sequential()
	model.add(Dense(256, input_shape=env.observation_space.shape, activation="relu"))
	model.add(Dense(512, activation="relu"))
	model.add(Dense(256, activation="relu"))
	model.add(Dense(env.action_space.shape[0]))
	return model

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
def create_critic_model(env):
	state_input  = Input(shape=env.observation_space.shape)
	state_h1 = Dense(64, activation='relu')(state_input)
	state_h2 = Dense(128)(state_h1)
	#
	action_input = Input(shape=env.action_space.shape)
	action_h1    = Dense(128)(action_input)
	#
	merged = Add()([state_h2, action_h1])
	merged_h1 = Dense(128, activation='relu')(merged)
	final = Dense(1, activation='relu')(merged_h1)
	model = Model(input=[state_input,action_input], output=final)
	#
	adam  = Adam(lr=0.001)
	model.compile(loss="mse", optimizer=adam)
	return model

env = gym.make("Pendulum-v0")
actor_model  = create_actor_model(env)
critic_model = create_critic_model(env)

num_trials = 10000
trial_len  = 500

for _ in range(num_trials):
	cur_state = env.reset()
	action = env.action_space.sample()
	for step in range(trial_len):
		cur_state = cur_state.reshape(1, env.observation_space.shape[0])
		action = actor_model.predict(cur_state)
		action = action.reshape(1, env.action_space.shape[0])

		new_state, reward, done, _ = env.step(action)
		new_state = new_state.reshape(1, env.observation_space.shape[0])

		# reward associated with the original state is the same "Q learning formula"
		# as before, namely the current reward and the discounted future reward
		Q_prime = reward + critic_model.predict([cur_state,action])[0][0]
		critic_model.fit(cur_state, Q_prime)
		cur_state = new_state
		if done:
			break
	print("Step: {}".format(step))