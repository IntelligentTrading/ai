__author__ = 'AlexBioY'
import numpy as np
import importlib
import os
from collections import namedtuple

from src.data.data_sources import get_combined_cleaned_onecoin_df
from src.data.settings import  DATASET_TRANSFORM

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



# TODO: do it smarter (use keras function ot scipy) or use matrix multiplication
def _normalize_dataset(X):
    for example in range(X.shape[0]):
        X[example, :, 0] = (X[example, :, 0] - X[example, -1, 0]) / (np.max(X[example, :, 0]) - np.min(X[example, :, 0]))
        X[example, :, 1] = (X[example, :, 1] - X[example, -1, 1]) / (np.max(X[example, :, 1]) - np.min(X[example, :, 1]))
        X[example, :, 2] = (X[example, :, 2] - X[example, -1, 2]) / (np.max(X[example, :, 2]) - np.min(X[example, :, 2]))
        X[example, :, 3] = (X[example, :, 3] - X[example, -1, 3]) / (np.max(X[example, :, 3]) - np.min(X[example, :, 3]))
    return X


def one_coin_array_from_df(data_df, win_size, stride, label_func, num_classes, future, return_target,res_period):
    '''
    Transform an input ts into array [ examples, time points back fatures (LSTM modules), feature dimension ],
    Labels can be set to a different
    '''
    n = len(data_df)
    num_examples = int((n - win_size) / stride) # how many times we can srtide via the timeseries (number of possible examples)

    # (4968, 96, 1)
    predictors = data_df.shape[1]  # make prediction based on multivatiate ts, price and volume
    label_dummy_classes = num_classes

    data_set = np.zeros([num_examples, win_size, predictors])
    labels = np.zeros([num_examples, label_dummy_classes])

    # form training examples by shifting triugh the dataset
    logger.info("   One coin: Converting dataframe to dataset array,  " + str(num_examples) + " examples")
    for start_example in range(0, num_examples):
        end_example = start_example + win_size

        # build X array
        data_set[start_example, :, 0] = data_df[start_example:end_example]['price'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 1] = data_df[start_example:end_example]['volume'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 2] = data_df[start_example:end_example]['price_var'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 3] = data_df[start_example:end_example]['volume_var'].values.reshape([-1, 1])[:, 0]
        #TODO: add blockchain info here

        # assert X dimensions

        # get price for the prediction period and calculate its moments
        #prices = data_set[start_example, :, 0]
        future_prices = data_df[end_example:end_example + future]['price']

        #build X array
        module = importlib.import_module('src.data.datasets')
        func_obj = getattr(module, label_func)

        labels[start_example, :] = func_obj(future_prices, return_target)
        # assert the array dimencions

        if start_example % 10000 == 0:
            print("   ... processed examples: " + str(start_example))

    logger.info("   One coin: finished.")

    return data_set, labels


def label_3class_return_target(future_prices, return_target):
    '''
    calculate a dummy class number out of 90 future prices as 0 - same / 1 - up / 2 - down
    '''
    # 0 -same, 1-up, -1 -down
    label_dummy_classes=3

    open_price = future_prices[0]
    close_price = future_prices[-1]
    price_return = close_price - open_price
    percentage_return = 1 - (open_price - price_return) / open_price

    label = 0 if (abs(percentage_return) < return_target) else np.sign(percentage_return)

    dummy_labels = np.zeros([1,label_dummy_classes])

    # 0 - same / 1 - up / 2 - down
    if label == 0:
        dummy_labels[0, 0] = 1
    elif label == 1:
        dummy_labels[0, 1] = 1
    elif label == -1:
        dummy_labels[0, 2] = 1

    return dummy_labels


def label_2class_return_target(future_prices, return_target):
    # NOTE: return tagret  is ignored here

    # 1 - up, - 1 - down
    label_dummy_classes = 2

    open_price = future_prices[0]
    close_price = future_prices[-1]
    price_return = close_price - open_price
    percentage_return = 1 - (open_price - price_return) / open_price

    label = np.sign(percentage_return)

    dummy_labels = np.zeros([1, label_dummy_classes])

    # 0 - same / 1 - up / 2 - down
    if label == 1:
        dummy_labels[0, 0] = 1
    elif label == -1:
        dummy_labels[0, 1] = 1

    return dummy_labels


def get_dataset_manycoins_fused(COINS_LIST, db_name, ds_transform):
    '''
    Build the a full dataset X, Y by fusind all datasets of each coin from COIN_LIST
    - for each pair get ts of price and volume, calculate variance and build a df [time, price, vol, price_var, vol_var]
    - split this ts into pieces of win_size ad calculate a label for each
    - pile them up int one dataset
    '''

    ds_name = ds_transform
    res_period = DATASET_TRANSFORM[ds_transform].res_period

    # return from cache if files exists
    # TODO: CLEAN  the cache before real run!
    fname_x = 'X_'+ ds_name + '.pkl.npy'
    fname_y = 'Y_' + ds_name + '.pkl.npy'
    if os.path.isfile("data/processed/" + fname_x) and os.path.isfile("data/processed/" + fname_y):
        X = np.load("data/processed/" + fname_x)
        Y = np.load("data/processed/" + fname_y)
        logger.info("    ... got X, Y datasets from cache:")
        logger.info("        Y Datasets: same= " + str(sum(Y[:, 0])) + ' | UP= ' + str(sum(Y[:, 1])) + ' | DOWN= ' + str(sum(Y[:, 2])))
        return X, Y

    X = []  # (147319, 200, 4) - 4 is price, volume, price_var, volume_var
    Y = []  # (147319, 3)  - 3 is number of classes

    logger.info(" > Form data set X array from a coin list:" + str(COINS_LIST))

    for transaction_coin, counter_coin in COINS_LIST:
        # retrieve a time series df from DB as [time,price,volume, price_var, volume_var]
        data_df = get_combined_cleaned_onecoin_df(db_name, transaction_coin, counter_coin, res_period)

        # convert this df into a array of shape of (147319, 200, 4) = (examples, time_back, features)
        X_train_one, Y_train_one = one_coin_array_from_df(
            data_df=data_df,
            **DATASET_TRANSFORM[ds_transform]._asdict()  # all parameters of data transformation are in data.settings
        )
        del data_df

        # pile up into one array
        if X == []:
            X = X_train_one
            Y = Y_train_one
        else:
            X = np.concatenate((X, X_train_one), axis=0)
            Y = np.concatenate((Y, Y_train_one), axis=0)


    # delete all examples with NaN inside
    idx2delete = []
    for n in range(X.shape[0] - 1):
        if np.isnan(X[n, :, :]).any():
            idx2delete.append(n)
    X = np.delete(X, (idx2delete), axis=0)
    Y = np.delete(Y, (idx2delete), axis=0)

    logger.info("> X,Y Datasets have been built: same= " + str(sum(Y[:,0])) + ' | UP= ' + str(sum(Y[:,1])) + ' | DOWN= ' + str(sum(Y[:,2])))

    # TODO: shaffle dataset

    # normalize
    # TODO: can I do it in-place?
    X = _normalize_dataset(X)

    # sanity check
    logger.info("   ... Sanity Checking for NaN in dataset: check for any nan")
    for n in range(X.shape[0]):
        if np.isnan(X[n, :, :]).any():
            logger.info(n)

    logger.info("final X dataset shape: " + str(X.shape))
    logger.info("final Y dataset shape: " + str(Y.shape))

    # TODO: check if folder is exists
    np.save("data/processed/"+fname_x, X)
    np.save("data/processed/"+fname_y, Y)

    return X, Y
