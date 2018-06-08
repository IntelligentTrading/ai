import time
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
#from keras.initializers import glorot_uniform
#from keras.utils import to_categorical
from keras.optimizers import Adam, adagrad
from keras import backend as K


def lstm_model():
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()

    model.add(
        LSTM(
            90,
            return_sequences=True,
            input_shape=(timesteps, data_dim),
            dropout=0.15
        )
    )  # returns a sequence of vectors of dimension 32

    model.add(LSTM(64, return_sequences=True, dropout=0.15))

    model.add(LSTM(32, return_sequences=True, dropout=0.15))

    model.add(LSTM(16, dropout=0.15))  # return a single vector of dimension 32

    model.add(Dense(num_classes, activation='softmax'))


    optimizer = adagrad(lr=0.0005)

    start = time.time()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    print("> Compilation Time : ", time.time() - start)
    return model