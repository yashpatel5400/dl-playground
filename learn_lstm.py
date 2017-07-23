import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def learn_lstm():
	np.random.seed(7)
	df = pd.read_csv("international-airline-passengers.csv",
	usecols=[1], skipfooter=3)
	dataset = df.values.astype('float32')
	scaler = MinMaxScaler(feature_range=(0,1))
	dataset = scaler.fit_transform(dataset)

	train_prop = .67
	training_samples = int(len(dataset) * train_prop)
	training_set = dataset[:training_samples]
	test_set = dataset[training_samples+1:]

	look_back = 1
	trainX, trainY = make_forward_dataset(training_set, look_back=look_back)
	testX, testY   = make_forward_dataset(test_set, look_back=look_back)

	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss="mean_squared_error", 
	optimizer="adam")
	model.fit(trainX, trainY, epochs=100, batch_size=1)

	predictTrain = model.predict(trainX)
	predictTest  = model.predict(testX)

	trainAcc = np.sum(predictTrain == trainY)/len(trainY)
	testAcc  = np.sum(predictTest == testY)/len(testY)

	print("Train accuracy: {}".format(trainAcc))
	print("Test accuracy: {}".format(testAcc))

def make_forward_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		dataX.append(dataset[i:(i+look_back), 0])
		dataY.append(dataset[i+look_back, 0])
	return np.array(dataX), np.array(dataY)