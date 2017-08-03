"""
developing a chatbot in Keras
"""

import tensorflow as tf
import numpy as np

from keras.models import Sequential, Model
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import np_utils

def create_dataset(dataset, look_back=1):
	training_set_X, training_set_Y = [], []
	for i in range(len(dataset)-look_back):
		training_set_X.append(dataset[i:i+look_back])
		training_set_Y.append(dataset[i+look_back])
	return training_set_X, training_set_Y

def keras_sequential_model(vocab_size, look_back=1, learning_rate=0.001):
	model = Sequential()
	model.add(LSTM(256, input_shape=(look_back, 1), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256))
	model.add(Dropout(0.2))
	model.add(Dense(vocab_size))
	adam = Adam(lr=learning_rate)
	model.compile(loss="categorical_crossentropy",
		optimizer=adam)
	return model

def keras_functional_model(vocab_size, look_back=1, learning_rate=0.001):
	inputs = Input(shape=(look_back,))
	h1 = LSTM(24, return_sequences=True)(inputs)
	h2 = LSTM(24, return_sequences=True)(h1)
	h3 = Dense(48)(h2)
	outputs = Dense(vocab_size)(h3)
	model = Model(inputs=inputs, outputs=outputs)
	adam = Adam(lr=learning_rate)
	model.compile(loss="categorical_cross_entropy",
		optimizer=adam)
	return model

# model/training parameters
look_back  = 8
batch_size = 32

# preprocessing
train_text = np.array(open("fab.txt").read().strip().split())
unique_words = np.unique(train_text)
num_unique   = len(unique_words)
word_to_num = dict(zip(unique_words, range(num_unique)))
num_to_word = dict(zip(range(num_unique), unique_words))

train_text_embed = [word_to_num[word] for word in train_text]
training_set_X, training_set_Y = create_dataset(
	train_text_embed, look_back=look_back)

training_set_X = np.array(training_set_X)
training_set_X = training_set_X.reshape(training_set_X.shape[0], training_set_X.shape[1], 1)
training_set_X = training_set_X / float(unique_words)

training_set_Y = np.array(training_set_Y)
training_set_Y = np_utils.to_categorical(training_set_Y)

# model definitions
model = keras_sequential_model(num_unique, look_back=look_back)
model.fit(training_set_X, training_set_Y, 
	batch_size=batch_size, epochs=10)

model.save("test.model")

# seed input
temperature  = 0.45
gen_text_len = 45

seed_ind = np.random.randint(len(training_set_X))
seed = training_set_X[seed_ind]
seed = seed.reshape(1, seed.shape[0], seed.shape[1])
final_text = [x[0] for x in seed[0]]

for _ in range(gen_text_len):
	#if np.random.random() < temperature:
	#	predicted_value = np.random.randint(num_unique)
	#else: 
	#	predicted_weights = model.predict(seed)
	predicted_value = np.argmax(predicted_weights)
	final_text.append(predicted_value)
	seed = np.append(seed[:,1:,:], [[[ predicted_value / float(unique_words) ]]], axis=1)

final_str = " ".join([num_to_word[i] for i in final_text])
print(final_str)