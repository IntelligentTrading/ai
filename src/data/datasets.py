#TODO : use a Timeseries Generator of Keras : https://keras.io/preprocessing/sequence/

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


def df_to_X_onecoin(data_df, ds_transform):
    '''
    Transform an input ts into array [ examples, time points back fatures (LSTM modules), feature dimension ],
    Labels computing can be set to a different function
    '''
    win_size = DATASET_TRANSFORM[ds_transform].win_size
    stride = DATASET_TRANSFORM[ds_transform].stride
    label_func = DATASET_TRANSFORM[ds_transform].label_func
    num_classes = DATASET_TRANSFORM[ds_transform].num_classes
    future = DATASET_TRANSFORM[ds_transform].future
    res_period = DATASET_TRANSFORM[ds_transform].res_period

    n = len(data_df)
    if (n - win_size) < 0:
        logger.error("   DATASET is smaller then win_size! we need more data")
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
        data_set[start_example, :, 4] = data_df[start_example:end_example]['price_max'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 5] = data_df[start_example:end_example]['price_min'].values.reshape([-1, 1])[:, 0]
        #TODO: add blockchain info here


        # here we set the future values either to following proce values of price-max values (in case we predict max
        if label_func == 'label_3class_max_hit':
            future_values = data_df[end_example-1 : end_example + future]['price_max']
            threshold_1 = DATASET_TRANSFORM[ds_transform].threshold_1
            threshold_2 = DATASET_TRANSFORM[ds_transform].threshold_2
        elif label_func == 'label_3class_min_hit':
            future_values = data_df[end_example-1 : end_example + future]['price_min']
            threshold_1 = DATASET_TRANSFORM[ds_transform].threshold_1
            threshold_2 = DATASET_TRANSFORM[ds_transform].threshold_2
        elif label_func == 'label_2class_max_hit':
            future_values = data_df[end_example - 1: end_example + future]['price_min']
            threshold_1 = DATASET_TRANSFORM[ds_transform].threshold_1
            threshold_2 = None
        else:
            future_values = data_df[end_example-1 : end_example + future]['price'] # we also need the last price from example
            threshold_1 = DATASET_TRANSFORM[ds_transform].return_target
            threshold_2 = None

        #build Y array (labels)
        module = importlib.import_module('src.data.datasets')
        func_obj = getattr(module, label_func)

        labels[start_example, :] = func_obj(future_values, threshold_1, threshold_2)
        # assert the array dimencions

        if start_example % 3000 == 0:
            logger.info("   ... df->array examples completed: " + str(start_example))

    logger.info("   One coin: finished.")

    return data_set, labels




def label_3class_return_target(future_prices, threshold_1, threshold_2):
    '''
    calculate a dummy class number out of 90 future prices as 0 - same / 1 - up / 2 - down
    '''
    return_target = threshold_1
    # 0 -same, 1-up, 2 -down
    label_dummy_classes=3

    open_price = future_prices[0]
    close_price = future_prices[-1]
    price_return = close_price - open_price
    percentage_return = 1 - (open_price - price_return) / open_price

    label = 0 if (abs(percentage_return) < return_target) else np.sign(percentage_return)

    dummy_labels = np.zeros([1,label_dummy_classes]).astype(int)

    # 0 - same / 1 - up / 2 - down
    if label == 0:
        dummy_labels[0, 0] = 1
    elif label == 1:
        dummy_labels[0, 1] = 1
    elif label == -1:
        dummy_labels[0, 2] = 1

    return dummy_labels

def label_2class_return_target(future_prices, threshold_1, threshold_2):
    # NOTE: return tagret  is ignored here

    return_target = threshold_1
    # 1 - up, - 1 - down
    label_dummy_classes = 2

    open_price = future_prices[0]
    close_price = future_prices[-1]
    price_return = close_price - open_price
    percentage_return = 1 - (open_price - price_return) / open_price

    label = np.sign(percentage_return)

    dummy_labels = np.zeros([1, label_dummy_classes]).astype(int)

    # 0 - same / 1 - up / 2 - down
    if label == 1:
        dummy_labels[0, 0] = 1
    elif label == -1:
        dummy_labels[0, 1] = 1

    return dummy_labels

def label_3class_max_hit(future_prices, threshold_1, threshold_2):
    # 0 -same, 1-threshold_1, 2 -threshold_2
    label_dummy_classes=3

    open_price = future_prices[0]
    close_price = future_prices[-1]

    # min_price = np.min(future_prices)
    # percent_min = 1 - (open_price - min_price) / open_price
    max_price = np.max(future_prices)
    percent_max = (max_price-open_price) / open_price


    if  percent_max > threshold_2:
        label = 2
    elif percent_max > threshold_1:
        label = 1
    else:
        label = 0

    dummy_labels = np.zeros([1,label_dummy_classes]).astype(int)

    # 0 - same / 1 - up / 2 - down
    if label == 0:
        dummy_labels[0, 0] = int(1)
    elif label == 1:
        dummy_labels[0, 1] = int(1)
    elif label == 2:
        dummy_labels[0, 2] = int(1)

    return dummy_labels

def label_2class_max_hit(future_prices, threshold_1, threshold_2):
    # 0 -same, 1-threshold_1, 2 -threshold_2
    label_dummy_classes=2

    open_price = future_prices[0]
    close_price = future_prices[-1]

    # min_price = np.min(future_prices)
    # percent_min = 1 - (open_price - min_price) / open_price
    max_price = np.max(future_prices)
    percent_max = (max_price-open_price) / open_price

    if percent_max > threshold_1:
        label = 1
    else:
        label = 0

    dummy_labels = np.zeros([1,label_dummy_classes]).astype(int)

    # 0 - same / 1 - up / 2 - down
    if label == 0:
        dummy_labels[0, 0] = int(1)
    elif label == 1:
        dummy_labels[0, 1] = int(1)


    return dummy_labels


def combine_all_coins(COINS_LIST, db_name, ds_transform):
    '''
    Build the a full dataset X, Y by fusind all datasets of each coin from COIN_LIST
    - for each pair get ts of price and volume, calculate variance and build a df [time, price, vol, price_var, vol_var]
    - split this ts into pieces of win_size ad calculate a label for each
    - pile them up int one dataset
    '''

    res_period = DATASET_TRANSFORM[ds_transform].res_period

    X = []  # (147319, 200, 4) - 4 is price, volume, price_var, volume_var
    Y = []  # (147319, 3)  - 3 is number of classes

    logger.info(" > Form data set X array from a coin list:" + str(COINS_LIST))

    for transaction_coin, counter_coin in COINS_LIST:
        # retrieve a time series df from DB as [time,price,volume, price_var, volume_var]
        data_df = get_combined_cleaned_onecoin_df(db_name, transaction_coin, counter_coin, res_period)

        # convert this df into a array of shape of (147319, 200, 4) = (examples, time_back, features)
        # all parameters of data transformation are in data.settings
        X_train_one, Y_train_one = df_to_X_onecoin(data_df, ds_transform)
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

    if DATASET_TRANSFORM[ds_transform].num_classes == 3:
        logger.info("> X,Y Datasets have been built: same= " + str(sum(Y[:,0])) + ' | UP= ' + str(sum(Y[:,1])) + ' | DOWN= ' + str(sum(Y[:,2])))
    elif DATASET_TRANSFORM[ds_transform].num_classes == 2:
        logger.info("> X,Y Datasets have been built: UP= " + str(sum(Y[:, 0])) + ' | DOWN= ' + str( sum(Y[:, 1])) )
    else:
        logger.error("UNKNOWN NUMBER OF CLASSES!")

    # normalize
    # TODO: can I do it in-place?
    X = _normalize_dataset(X)

    # sanity check
    logger.info("   ... Sanity Checking for NaN in dataset: check for any nan")
    for n in range(X.shape[0]):
        if np.isnan(X[n, :, :]).any():
            logger.info(n)

    logger.info("=======> final X dataset shape: " + str(X.shape))
    logger.info("=======> final Y dataset shape: " + str(Y.shape))

    # # TODO: check if folder is exists
    # np.save("data/processed/"+fname_x, X)
    # np.save("data/processed/"+fname_y, Y)

    return X, Y

