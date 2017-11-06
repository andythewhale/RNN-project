import numpy as np                      # we need numpy

from keras.models import Sequential     # we need Sequential models, these are time series.
from keras.layers import Dense          # we need Dense layers for outputs
from keras.layers import LSTM           # we need LSTM layers for classification
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

    X = np.asarray(X)            # let's look at X as an array
    X.shape = (np.shape(X)[0:2]) # we only want the shape of X to follow from 0 --> 2
    y = np.asarray(y)            # let's look at y as an array
    y.shape = (len(y),1)         # we only want the shape of y to follow from its length <-- 1

    # return X, y
    return X,y


def build_part1_RNN(window_size):

    # let's build a model
    model = Sequential()                                    # let's id our model as sequential
    model.add(LSTM(13, input_shape = (window_size, 1)))     # let's use 13 LSTM units
    model.add(Dense(1))                                     # let's add a dense layer to determine out output
    return model                                            # let's return our model





def cleaned_text(text):

    # could just import this but ok.
    unwanted_characters = ['$', '%', '&', '*', '@', '\xa0', '¢', '¨', '©', 'ã']

    # get rid of unwanted characters

    for i in unwanted_characters:       # so for each item in unwanted characters
        text = text.replace(i, '')      # replace each unwanted character with nothing
    return text                         # return and replace text




def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    # we just need to cut it up again and append to input output,
    for i in range(window_size, len(text), step_size):      # so for each item in our range of those
        inputs.append(text[i-window_size])                  # we're going to want to look at each incoming letter
        outputs.append(text[i])                             # we're also going to want to create an outgoing letter


    return inputs, outputs

# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()                                                                # id as Sequential
    model.add(LSTM(200, input_shape = (window_size, num_chars)))                        # specify input shape and LSTMs
    model.add(Dense(num_chars, activation = 'softmax'))                                 # add dense output w/ softmax
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)   # add optimizer layer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)                 # add loss function
    return model                                                                        # return the model