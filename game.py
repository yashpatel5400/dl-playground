"""
basic game for learning reinforcement learning
"""

import numpy as np
import gym

# basic implementation

env = gym.make("CartPole-v0")
best_params = [0 for _ in range(4)]
max_steps = 0

for times in range(1000):
	observation = env.reset()
	params = np.random.random(4)
	for step in range(200): 
		action = int(np.dot(params, observation) > 0)
		observation, reward, done, info = env.step(action)
		if done:
			break
	if step == 199:
		break

print(times)

params = np.random.random(4)
observation = env.reset()
for _ in range(1000):
	action = int(np.dot(best_params, observation) > 0)
	observation, _, _, _ = env.step(action)
	env.render()

print("Completed {} steps".format(steps))

# hill climbing

env = gym.make("CartPole-v0")
best_params = np.random.rand(4) * 2 - 1
previous_best_steps = 0

for times in range(1000):
	observation = env.reset()
	delta = np.random.rand(4) * 2 - 1
	params = best_params + delta
	for step in range(200):
		action = int(np.dot(params, observation) > 0)
		observation, rewards, done, _ = env.step(action)
		if done:
			break
	if step == 199:
		break
	if step > previous_best_steps:
		previous_best_steps = step
		best_params = params

print(times)

# deep learning version

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import gym
from keras.models import load_model

env = gym.make("CartPole-v0")

min_score = 50
sim_steps = 500
trainingX, trainingY = [], []

# get data

scores = []
for _ in range(10000):
	observation = env.reset()
	score = 0
	training_sampleX, training_sampleY = [], []
	for step in range(sim_steps):
		action = np.random.randint(0, 2)
		# action corresponds to the previous observation so record before step
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

# define model

model = Sequential()
model.add(Dense(128, input_shape=(4,), activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.6))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(2, activation="softmax"))

model.compile(
	loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"])

# train model
model.fit(trainingX, trainingY, epochs=5)
scores = []
num_trials = 50
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