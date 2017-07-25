"""
learning how to apply RNN for word prediction
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def get_data():
	fn = "wonderland.txt"
	raw_data = open(fn, "r").read().lower()
	chars = list(set(raw_data))
	num_embed = [i for i in range(len(chars))]
	char_to_num = dict(zip(chars,num_embed))
	num_to_char = dict(zip(num_embed,chars))
	full_text = list(raw_data)
	converted = [char_to_num[char] for char in full_text]
	return converted, char_to_num, num_to_char

def create_datasets(dataset, look_back=1):
	X, Y = [], []
	for i in range(len(dataset) - look_back - 1):
		X.append(dataset[i:(i+look_back)])
		Y.append(dataset[i+look_back])
	return np.array(X), np.array(Y)

converted, char_to_num, num_to_char = get_data()
look_back = 100

trainX, trainY = create_datasets(converted, look_back=look_back)
num_chars = len(char_to_num.keys())
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))/float(num_chars)
trainY = np_utils.to_categorical(trainY)

model = Sequential()
model.add(LSTM(256, input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(66, activation="softmax"))
model.compile(loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"])

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(trainX, trainY, 
	epochs=5, verbose=1, batch_size=100, callbacks=callbacks_list)

# ----------------- generative stage --------------------------- #
fn = "weights-improvement-02-2.3004.hdf5"
model.load_weights(fn)
model.compile(loss="categorical_crossentropy",
	optimizer="adam",
	metrics=["accuracy"])

start = np.random.randint(0, len(trainX) - 1)
seed = trainX[start]

num_predict = 1000
seq = [num_to_char[int(num_chars *char[0])] for char in seed]
for _ in range(num_predict):
	next_probs = model.predict(seed.reshape(1, 100, 1))
	next_char_ind = np.argmax(next_probs)
	next_char = num_to_char[next_char_ind]
	seed = np.append(seed[1:], [next_char_ind/float(num_chars)])
	seq.append(next_char)

print("".join(seq))