import time
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras import metrics
from keras.optimizers import Adam, adagrad
from keras import backend as K


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


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
        metrics=['accuracy', metrics.categorical_accuracy]
    )
    print("> Compilation Time : ", time.time() - start)
    print(model.summary())

    return model

def predict_point_by_point(model, X):
    predicted = model.predict(X)
    return predicted