import time
from artemis.experiments import ExperimentFunction

from vizualization.plotting import plot_model_results, plot_3class_colored_prediction
from data.data_sources import get_combined_cleaned_onecoin_df
from data.datasets import get_dataset_fused
from models.keras_models import build_lstm_model, Metrics

import tensorflow as tf
import keras
from keras import backend as K


def display_train_result(results):
    plot_model_results(results)

def compare_trainings(dict_of_histories):
    print("you can add a training comparison here to show it in UI")


@ExperimentFunction(display_function=display_train_result)
def single_train( res_period, win_size, future, return_target, label_func, data_dim, num_classes, lr, batch_size, epochs):
    # list all coin pairs for the training set
    TRAIN_COINS_LIST = [('BTC', 2)]
    # TRAIN_COINS_LIST = [
    #     ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    #     ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0)
    # ]


    lstm_layers = [
        {'layer':'input', 'units':50, 'dropout':0.15},
        {'layer':'l2',    'units':25, 'dropout':0.15},
        #{'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.15}
    ]

    # lstm_layers = [
    #     {'layer':'input', 'units':90, 'dropout':0.15},
    #     {'layer':'l2',    'units':64, 'dropout':0.15},
    #     {'layer':'l3',    'units':32, 'dropout':0.15},
    #     {'layer':'last',  'units':16, 'dropout':0.1}
    # ]

    # build a dataset for training
    db_name = 'postgre_stage'   # 'prodcopy',
    print("=========== Form a TEST data set =========== ")
    X_train, Y_train = get_dataset_fused(
        COINS_LIST=TRAIN_COINS_LIST,
        db_name=db_name,
        res_period=res_period,
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func,
        num_classes=num_classes
    )

    # set a validation ts (BTC/2 here, can be changed)
    print("=========== Form a VALIDATION data set (BTC) =========== ")
    VALID_COIN = 'BTC'
    VALID_COUNTER = 2

    # get validation price
    raw_valid_data_df = get_combined_cleaned_onecoin_df(db_name=db_name, transaction_coin=VALID_COIN, counter_coin=VALID_COUNTER, res_period=res_period)
    raw_validation_price = raw_valid_data_df['price'].values

    # get validation dataset for futher metrics
    X_valid, Y_valid = get_dataset_fused(
        COINS_LIST=[(VALID_COIN,VALID_COUNTER)],
        db_name='prodcopy',
        res_period=res_period,
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func,
        num_classes=num_classes
    )

    # build a model
    model = build_lstm_model(win_size, data_dim, num_classes, lstm_layers, lr)

    # train the model
    print("=========== TRAINING  ============ ")
    metrics = Metrics()

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15,
        callbacks=[metrics],
        verbose = 2         # 0 = silent, 1 = progress bar, 2 = one line per epoch
    )

    #plot_model_metrics(history)

    model.save("models/lstm_model.h5")

    ### plot colored prediction on train data
    # get
    point=2000
    print("===========  PREDICTING on validation dataset  ==============")
    start = time.time()
    y_predicted_valid = model.predict(X_valid)

    plot_kvargs = {
        'price': raw_validation_price,
        'y_predicted': y_predicted_valid,
        'point': point,
        'win_size': win_size,
        'future': future
    }
    plot_3class_colored_prediction(**plot_kvargs)
    print("Prediction Time : ", time.time() - start)

    #close keras session
    K.clear_session()

    return history.history, metrics.get_scores(), plot_kvargs


def add_all_experiments_variants():
    '''
    add all variants of the model tp train
    so if we need to train several models with some sligtly different parameters, we add all of them here
    '''
    # parameters of the dataset and model
    res_period = '10min'
    win_size = 288  # 48h back
    future = 24  # 4h forward
    return_target = 0.01  # need to tune it to make classes balanced
    data_dim = 4  # price, price_var, volume, volume_var

    label_func = 'label_3class_return_target'
    num_classes = 3  # up, same, down

    #lr = 0.0005
    batch_size = 512  # might be up to 7000 if enough memory and GPU
    epochs = 1

    ###############################
    # you can give to an experiment your own name
    # my_experiment_function.add_variant('big_a', a=10000)

    single_train.add_variant(
        'basic model 1 batch',
        res_period=res_period,
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func,
        data_dim=data_dim,
        num_classes=num_classes,
        lr=0.0006,
        batch_size=batch_size,
        epochs=epochs
    )

    # single_train.add_variant(
    #     res_period=res_period,
    #     win_size=win_size,
    #     future=future,
    #     return_target=return_target,
    #     label_func=label_func,
    #     data_dim=data_dim,
    #     num_classes=num_classes,
    #     lr=0.01,
    #     batch_size=batch_size,
    #     epochs=epochs
    # )


#TODO
# add a lot of performance measures
# write a description MD file
# add Logger

# save model file in the same experiment folder - do it better!!



if __name__ == '__main__':
    print("   TensorFlow = " + tf.__version__)
    print("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    # we can add a single experiment to artemis by single_train
    # or run add_all_experiments_variants() to add a numebr of experiments all together
    add_all_experiments_variants()

    # by uncommenting and running this we can browse and run experiment from command line on the server
    #single_train.browse()

    # or run automatically one by onw here
    variants = single_train.get_all_variants()
    experiment_1 = variants[1]
    experiment_1.run()

    # Try
    #   run all
    #   compare all
    #   display 1
    # view full           View all columns (the default view)
    # delete 4.1          Delete record 1 of experiment 4
    # delete unfinished   Delete all experiment records that have not run to completion
    # delete 4-6          Delete all records from experiments 4, 5, 6.  You will be asked to confirm the deletion.




###########################################################

# TODO:
# - leave what I have as price_direction_indicator, with two classes and make them balanced (undersampling? adjusting delta?)
#    - so we can build a signal as indicator changed from - to +
# - add one more ANN which is a signal by itself, with anomaly detection approach with NN

# - try to learn for market manipulation detection.. BTC is clearly manipulated, try to detect this event (Orders? )

# - try to add ETh to BTC as a predictor... may be it works as a predictor


