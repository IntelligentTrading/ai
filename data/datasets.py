import pandas as pd
import numpy as np
from data.data_sources import ittconnection
from data.data_sources import get_raw_price, get_raw_volume


# TODO: do it smarter (use keras function ot scipy) or use matrix multiplication
def _normalize_dataset(X):
    for example in range(X_train.shape[0]):
        X[example, :, 0] = (X[example, :, 0] - X[example, -1, 0]) / (np.max(X[example, :, 0]) - np.min(X[example, :, 0]))
        X[example, :, 1] = (X[example, :, 1] - X[example, -1, 1]) / (np.max(X[example, :, 1]) - np.min(X[example, :, 1]))
        X[example, :, 2] = (X[example, :, 2] - X[example, -1, 2]) / (np.max(X[example, :, 2]) - np.min(X[example, :, 2]))
        X[example, :, 3] = (X[example, :, 3] - X[example, -1, 3]) / (np.max(X[example, :, 3]) - np.min(X[example, :, 3]))
    return X


def build_dataset_array_from_df(data_df, win_size, stride, label_func, future, return_target):
    '''
    Transform an input ts into array [ examples, time points back fatures (LSTM modules), feature dimension ],
    Labels can be set to a different
    '''
    n = len(data_df)
    num_examples = int((n - win_size) / stride) # how many times we can srtide via the timeseries (number of possible examples)

    # (4968, 96, 1)
    predictors = data_df.shape[1]  # make prediction based on multivatiate ts, price and volume

    data_set = np.zeros([num_examples, win_size, predictors])
    labels = np.zeros([num_examples, label_dummy_classes])
    prices = np.zeros([num_examples, 1])

    # form training examples by shifting triugh the dataset
    for start_example in range(0, num_examples):
        end_example = start_example + win_size

        # build X array
        data_set[start_example, :, 0] = data_df[start_example:end_example]['price'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 1] = data_df[start_example:end_example]['volume'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 2] = data_df[start_example:end_example]['price_var'].values.reshape([-1, 1])[:, 0]
        data_set[start_example, :, 3] = data_df[start_example:end_example]['volume_var'].values.reshape([-1, 1])[:, 0]
        #TODO: ad blockchain info here

        # assert X dimensions

        # get price for the prediction period and calculate its moments
        prices = data_set[start_example, :, 0]  #todo     prices[start_example, :] = open_price
        future_prices = data_df[end_example:end_example + future]['price']

        #build X array
        labels[start_example, :] = label_func(future_prices, return_target)
        # assert the array dimencions


    return data_set, labels, prices



def label_3class_return_target(future_prices, return_target):
    '''
    calculate a dumy class number out of 90 future prices as 0 - same / 1 - up / 2 - down
    '''

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
    pass

def get_dataset_fused(COINS_LIST, db_name, res_period = '10min', win_size, future, return_target, label_func):
    '''
    Build the a full dataset X, Y by fusind all datasets of each coin from COIN_LIST
    - for each pair get ts of price and volume, calculate variance and build a df [time, price, vol, price_var, vol_var]
    - split this ts into pieces of win_size ad calculate a label for each
    - pile them up int one dataset
    '''
    db_connection = ittconnection(db_name)

    X = []  # (147319, 200, 4) - 4 is price, volume, price_var, volume_var
    Y = []  # (147319, 3)  - 3 is number of classes

    for transaction_coin, counter_coin in COINS_LIST:
        # get raw ts from DB
        raw_price_ts = get_raw_price(db_connection, transaction_coin, counter_coin)
        raw_volume_ts = get_raw_volume(db_connection, transaction_coin, counter_coin)

        # merge because the timestamps must match, and merge left because price shall have a priority
        raw_data_frame = pd.merge(raw_price_ts, raw_volume_ts, how='left', left_index=True, right_index=True)
        print('Shape of ' + transaction_coin + str(raw_data_frame.shape))
        raw_data_frame[pd.isnull(raw_data_frame)] = None

        # add variance, resample (for smoothing)
        data_ts = raw_data_frame.resample(rule=res_period).mean()
        data_ts['price_var'] = raw_data_frame['price'].resample(rule=res_period).var()
        data_ts['volume_var'] = raw_data_frame['volume'].resample(rule=res_period).var()
        data_ts = data_ts.interpolate()

        # convert this df into a array of shape of (147319, 200, 4) = (examples, time_back, features)
        X_train_one, Y_train_one, y_tr_price = build_dataset_array_from_df(
            data_df=data_ts,
            win_size=win_size,
            stride=1,
            label_func=label_func,
            future=future,
            return_target=delta
        )

        # pile up into one array
        if X_train == []:
            X_train = X_train_one
            Y_train = Y_train_one
        else:
            X_train = np.concatenate((X_train, X_train_one), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_one), axis=0)

        del raw_price_ts, raw_volume_ts, raw_data_frame

    db_connection.close()

    # delete all examples with NaN inside
    idx2delete = []
    for n in range(X_train.shape[0] - 1):
        if np.isnan(X_train[n, :, :]).any():
            idx2delete.append(n)
    X_train = np.delete(X_train, (idx2delete), axis=0)
    Y_train = np.delete(Y_train, (idx2delete), axis=0)

    # TODO: shaffle dataset

    # normalize
    # TODO: can I do it in-place?
    X_train = _normalize_dataset(X_train)

    # sanity check
    for n in range(X_train.shape[0]):
        if np.isnan(X_train[n, :, :]).any():
            print(n)

    print(X_train.shape)
    print(Y_train.shape)
