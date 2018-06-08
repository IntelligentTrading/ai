import pandas as pd
import numpy as np


# REGRESSION: dataset for regression: split one continues timeseries inro many subsets by striding
def ts_to_array_dataset_regression(data_df, win_size, stride, future, label_type):
    n = len(data_df)
    num_examples = int((n - win_size) / stride)

    # (4968, 96, 1)
    predictors = data_df.shape[1]  # make prediction based on multivatiate ts, price and volume

    data_set = np.zeros([num_examples, win_size, predictors])
    labels = np.zeros([num_examples, 1])
    prices = np.zeros([num_examples, 1])

    for ex in range(0, num_examples):
        one_example_0 = data_df[ex:ex + win_size]['price'].values.reshape([-1, 1])
        one_example_1 = data_df[ex:ex + win_size]['volume'].values.reshape([-1, 1])
        one_example_2 = data_df[ex:ex + win_size]['price_var'].values.reshape([-1, 1])
        one_example_3 = data_df[ex:ex + win_size]['volume_var'].values.reshape([-1, 1])
        last_price = one_example_0[-1, 0]

        data_set[ex, :, 0] = one_example_0[:, 0]
        data_set[ex, :, 1] = one_example_1[:, 0]
        data_set[ex, :, 2] = one_example_2[:, 0]
        data_set[ex, :, 3] = one_example_3[:, 0]

        future_prices = data_df[ex + win_size:ex + win_size + future]['price'].values
        open_price = future_prices[0]
        close_price = future_prices[-1]
        price_return = close_price - open_price
        percentage_return = 1 - (last_price - price_return) / last_price

        if label_type == "price":
            label = open_price

        if label_type == "return":
            label = price_return

        delta = 0.03
        if label_type == "percent_return":
            label = 0 if (abs(percentage_return) < delta) else np.sign(percentage_return)

        labels[ex, :] = label
        prices[ex, :] = open_price

    return data_set, labels, prices


def ts_to_array_dataset_classification(data_df, win_size, stride, future, delta):
    n = len(data_df)
    num_examples = int((n - win_size) / stride)

    # (4968, 96, 1)
    predictors = data_df.shape[1]  # make prediction based on multivatiate ts, price and volume

    data_set = np.zeros([num_examples, win_size, predictors])
    labels = np.zeros([num_examples, 3])
    prices = np.zeros([num_examples, 1])

    for ex in range(0, num_examples):
        one_example_0 = data_df[ex:ex + win_size]['price'].values.reshape([-1, 1])
        one_example_1 = data_df[ex:ex + win_size]['volume'].values.reshape([-1, 1])
        one_example_2 = data_df[ex:ex + win_size]['price_var'].values.reshape([-1, 1])
        one_example_3 = data_df[ex:ex + win_size]['volume_var'].values.reshape([-1, 1])
        last_price = one_example_0[-1, 0]

        future_prices = data_df[ex + win_size:ex + win_size + future]['price'].values
        open_price = future_prices[0]
        close_price = future_prices[-1]
        price_return = close_price - open_price
        percentage_return = 1 - (last_price - price_return) / last_price

        label = 0 if (abs(percentage_return) < delta) else np.sign(percentage_return)

        data_set[ex, :, 0] = one_example_0[:, 0]
        data_set[ex, :, 1] = one_example_1[:, 0]
        data_set[ex, :, 2] = one_example_2[:, 0]
        data_set[ex, :, 3] = one_example_3[:, 0]

        prices[ex, :] = open_price

        # 0 - same / 1 - up / 2 - down
        if label == 0:
            labels[ex, 0] = 1
        elif label == 1:
            labels[ex, 1] = 1
        elif label == -1:
            labels[ex, 2] = 1

    return data_set, labels, prices


def normalize_dataset(X):
    for example in range(X_train.shape[0]):
        X_train[example, :, 0] = (X_train[example, :, 0] - X_train[example, -1, 0]) / (
                    np.max(X_train[example, :, 0]) - np.min(X_train[example, :, 0]))
        X_train[example, :, 1] = (X_train[example, :, 1] - X_train[example, -1, 1]) / (
                    np.max(X_train[example, :, 1]) - np.min(X_train[example, :, 1]))
        X_train[example, :, 2] = (X_train[example, :, 2] - X_train[example, -1, 2]) / (
                    np.max(X_train[example, :, 2]) - np.min(X_train[example, :, 2]))
        X_train[example, :, 3] = (X_train[example, :, 3] - X_train[example, -1, 3]) / (
                    np.max(X_train[example, :, 3]) - np.min(X_train[example, :, 3]))


def



