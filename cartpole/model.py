"""
__name__   = model.py
__author__ = Yash Patel
__description__ = Defines model to be trained on the Cartpole data,
predicting the directioal action to take given 4D observation state
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout

def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))
    
    model.add(Dense(512, activation="relu"))
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
    return model

if __name__ == "__main__":
    model = create_model()