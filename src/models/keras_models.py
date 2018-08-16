import numpy as np
#from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
#from sklearn.exceptions import UndefinedMetricWarning

from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras import metrics
from keras.optimizers import Adam, adagrad
from keras import backend as K
from keras.callbacks import Callback

import warnings

np.set_printoptions(precision=3)

# I have to implement it from scratch because AWS image crashes with scipy metrics for some reason
def f1_scores(y_true, y_predicted):

    tp = (y_true * y_predicted).sum(axis=0)
    fp = y_predicted.sum(axis=0) - tp
    fn = y_true.sum(axis=0) - tp

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)  # how accurate we predict every positives ( fn - those who shall be predicted but they are not
    f1 = 2*((precision*recall)/(precision+recall))
    return [f1, precision, recall]


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        _val_f1, _val_precision, _val_recall = f1_scores(y_true=val_targ, y_predicted=val_predict)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('      + val_f1: %s || val_precision: %s || val_recall: %s'  % (str(_val_f1), str(_val_precision), str(_val_recall)))
        return

    def get_scores(self):
        return {'f1':self.val_f1s, 'recall':self.val_recalls, 'precision':self.val_precisions}



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


    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'] #, metrics.categorical_accuracy]
    )

    return model
