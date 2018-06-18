import time
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
#from keras.initializers import glorot_uniform
#from keras.utils import to_categorical
from keras.optimizers import Adam, adagrad
from keras import backend as K


def build_lstm_model(win_size_timesteps, data_dim,num_classes, layers_dict, lr):
    # expected input data shape: (batch_size, timesteps, data_dim)

    model = Sequential()

    for layer in layers_dict:
        if layer['layer'] == 'input':
            model.add(LSTM(layer['units'], return_sequences=True, input_shape=(win_size_timesteps, data_dim), dropout=layer['dropout']))
        elif layer['layer'] == 'last':
            model.add(LSTM(layer['units'], dropout=layer['dropout']))  # return a single vector of dimension 32
        else:
            model.add(LSTM(layer['units'], return_sequences=True, dropout=layer['dropout']))

    model.add(Dense(num_classes, activation='softmax'))

    optimizer = adagrad(lr)

    start = time.time()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    print("> Compilation Time : ", time.time() - start)
    print(model.sumary())

    return model