import time
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras import metrics
from keras.optimizers import Adam, adagrad
from keras import backend as K

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(y_true=val_targ, y_pred=val_predict, average=None)   # can change averate to weitherd?
        _val_recall = recall_score(y_true=val_targ, y_pred=val_predict, average=None)
        _val_precision = precision_score(y_true=val_targ, y_pred=val_predict, average=None)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(' — val_f1: %s — val_precision: %s — val_recall %s'  % (str(_val_f1), str(_val_precision), str(_val_recall)))
        return




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
        metrics=['accuracy'] #, metrics.categorical_accuracy]
    )
    #print("> Compilation Time : ", time.time() - start)
    #print(model.summary())

    return model
