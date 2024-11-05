from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def build_model(input_shape, action_space):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    return model
