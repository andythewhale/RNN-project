import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    # as the example in the notebook shows we need to slice things.
    # but we need to use a more generalized method.

    # so for each item in our window_size (I had to add window size here,
    # i did not realize that without it we would evaluate too many indices).
    for i in range(window_size, len(series)):

        # just append the current distance subtracted by the
        # total distance needed to travel for x
        # also append the series at the current position for y
        X.append(series[i - window_size:i])
        y.append(series[i])


    # this should still be good to go
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):

    # let's build a model
    model = Sequential()
    model.add(LSTM(13, input_shape= (window_size, 1)))
    model.add(Dense(1))
    return model




### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass
