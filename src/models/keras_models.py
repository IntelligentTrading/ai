import numpy as np
import time

from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras import metrics
from keras.optimizers import Adam, adagrad
from keras import backend as K
from keras.callbacks import Callback

from src.vizualization.plotting import plot_model_results, plot_3class_colored_prediction
from src.data.data_sources import get_combined_cleaned_onecoin_df
from src.data.datasets import get_dataset_manycoins_fused, one_coin_array_from_df
from src.data.settings import DATASET_TRANSFORM

from artemis.experiments import ExperimentFunction

import warnings
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.set_printoptions(precision=3)

# I have to implement it from scratch because AWS image crashes with scipy metrics for some reason
def f1_scores(y_true, y_predicted):

    tp = ((y_true * y_predicted).sum(axis=0)).astype(int)
    fp = (y_predicted.sum(axis=0) - tp).astype(int)
    fn = (y_true.sum(axis=0) - tp).astype(int)
    print("    tp = %s, fp = %s, fn = %s" % (str(tp), str(fp),str(fn)))

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
        #val_predict = (self.model.predict(self.validation_data[0])).round()  # this is if we have softmax,
        softmax_tmp = self.model.predict(self.validation_data[0])
        val_predict = (softmax_tmp == softmax_tmp.max(axis=1)[:,None]).astype(int)  # convert the max row to one, others to zero
        val_targ = self.validation_data[1]

        _val_f1, _val_precision, _val_recall = f1_scores(y_true=val_targ, y_predicted=val_predict)

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('      +[same,up,down]: val_f1: %s || val_precision: tp/(tp+fp) : %s || val_recall: tp/(tp+fn) : %s'  % (str(_val_f1), str(_val_precision), str(_val_recall)))
        return

    def get_scores(self):
        return {'f1':self.val_f1s, 'recall':self.val_recalls, 'precision':self.val_precisions}


def compare_trainings(dict_of_histories):
    print("you can add a training comparison here to show it in UI")


def display_train_result(results):
    plot_model_results(results)


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


@ExperimentFunction(display_function=display_train_result,  is_root=True)
def rnn_train_basic(
        train_coin_list=[('BTC', 2), ('ETH', 0), ('LTC',2), ('XRP', 0)],
        ds_transform='basic_10m_288_24_3class_return0.01',
        lstm_layers=[],
        lr=0.0008,
        batch_size=1024,
        epochs = 3):

    # TODO get data_dim and num_classes from 'label_3class_return_target'
    data_dim = 4  # price, price_var, volume, volume_var
    win_size = DATASET_TRANSFORM[ds_transform].win_size
    num_classes = DATASET_TRANSFORM[ds_transform].num_classes
    res_period = DATASET_TRANSFORM[ds_transform].res_period
    future = DATASET_TRANSFORM[ds_transform].future


    # build a dataset for training
    db_name = 'postgre_stage'   # 'prodcopy',
    logger.info(">>>>>>>>>>>>>> Build a TRAINING data set ")
    X_train, Y_train = get_dataset_manycoins_fused(COINS_LIST=train_coin_list, db_name=db_name, ds_transform=ds_transform)

    # build a model
    model = build_lstm_model(win_size, data_dim, num_classes, lstm_layers, lr)

    # train the model
    logger.info(">>>>>>>>>>>>>>>>>> START TRAINING  ")
    metrics = Metrics()  # define custom metrics for keras

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[metrics],
        verbose = 2,         # 0 = silent, 1 = progress bar, 2 = one line per epoch
        shuffle=True,
        class_weight={0:1., 1:10., 2:10.}
    )

    model.save("models/lstm_" + ds_transform + ".h5")
    print(model.summary())

    ################## Check trained model on an independent validation dataset (BTC,2)
    # TODO: check prediction on more datasets, like ETC etc

    logger.info(">>>>>>>>> Build a VALIDATION data set (BTC) ")
    VALID_COIN = 'BTC'
    VALID_COUNTER = 2

    # get price for validation coin
    logger.info("   :: get BTC price ts for plotting:")
    raw_valid_data_df = get_combined_cleaned_onecoin_df(db_name=db_name, transaction_coin=VALID_COIN, counter_coin=VALID_COUNTER, res_period=res_period)
    raw_validation_price = raw_valid_data_df['price'].values

    # get validation dataset for futher metrics
    logger.info("   :: build X,Y for validation dataset:")
    X_valid, Y_valid = get_dataset_manycoins_fused([(VALID_COIN,VALID_COUNTER)], db_name, ds_transform=ds_transform)

    ### plot colored prediction on train data
    # get
    point=2500
    logger.info(">>>>>>>>>>  PREDICTING and PLOTTING on validation dataset (BTC)")
    start = time.time()
    y_predicted_valid = model.predict(X_valid)
    logger.info(" Prediction completed in : " + str(time.time() - start))

    #show f1 score on validation data
    val_predict = (y_predicted_valid == y_predicted_valid.max(axis=1)[:, None]).astype(int)
    _val_f1, _val_precision, _val_recall = f1_scores(y_true=Y_valid, y_predicted=val_predict)
    logger.info('  Valid on BTC [same,up,down] =  F1: %s || PRECISION: tp/(tp+fp) : %s || RECALL: tp/(tp+fn) : %s' % (str(_val_f1), str(_val_precision), str(_val_recall)))

    plot_kvargs = {
        'price': raw_validation_price,
        'y_predicted': y_predicted_valid,
        'point': point,
        'win_size': win_size,
        'future': future,
        'y_true': Y_valid
    }


    #close keras session
    K.clear_session()

    return history.history, metrics.get_scores(), plot_kvargs, model.get_config(), {'f1':_val_f1, 'precision':_val_precision, 'recall':_val_recall}
