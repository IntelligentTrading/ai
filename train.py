__author__ = 'AlexY'

import shutil
import time
from artemis.experiments import ExperimentFunction
from src.vizualization.plotting import plot_model_results, plot_3class_colored_prediction
from src.data.data_sources import get_combined_cleaned_onecoin_df
from src.data.datasets import get_dataset_manycoins_fused
from src.models.keras_models import build_lstm_model, Metrics

import tensorflow as tf
import keras
from keras import backend as K

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def display_train_result(results):
    plot_model_results(results)


def compare_trainings(dict_of_histories):
    print("you can add a training comparison here to show it in UI")


TRAIN_COINS_LIST = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0)
]


@ExperimentFunction(display_function=display_train_result,  is_root=True)
def rnn_1_train_basic(
        train_coin_list=[('BTC', 2), ('ETH', 0)],
        res_period='10min',
        win_size=288,
        future=24,
        return_target=0.01,
        label_func='label_3class_return_target',
        num_classes=3,
        lr=0.0008,
        batch_size=1024,
        epochs = 3):

    data_dim = 4  # price, price_var, volume, volume_var


    lstm_layers = [
        {'layer':'input', 'units':90, 'dropout':0.15},
        {'layer':'l2',    'units':64, 'dropout':0.15},
        {'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.1}
    ]

    # build a dataset for training
    db_name = 'postgre_stage'   # 'prodcopy',
    logger.info("=========== Form a TRAINING data set =========== ")
    X_train, Y_train = get_dataset_manycoins_fused(
        COINS_LIST=train_coin_list,
        db_name=db_name,
        res_period=res_period,
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func,
        num_classes=num_classes
    )

    # set a validation ts (BTC/2 here, can be changed)
    logger.info("=========== Form a VALIDATION data set (BTC) =========== ")
    VALID_COIN = 'BTC'
    VALID_COUNTER = 2

    # get validation price
    logger.info("   need price for plotting:")
    raw_valid_data_df = get_combined_cleaned_onecoin_df(db_name=db_name, transaction_coin=VALID_COIN, counter_coin=VALID_COUNTER, res_period=res_period)
    raw_validation_price = raw_valid_data_df['price'].values

    # get validation dataset for futher metrics
    logger.info("   get X,Y for validation dataset:")
    X_valid, Y_valid = get_dataset_manycoins_fused(
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
    logger.info("=========== TRAINING  ===================== ")
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

    # TODO: name convention... or put it into artemis folder
    model.save("models/lstm_model.h5")

    ### plot colored prediction on train data
    # get
    point=2000
    logger.info("===========  PREDICTING on validation dataset  ==============")
    start = time.time()
    y_predicted_valid = model.predict(X_valid)
    logger.info("Prediction Time : " + str(time.time() - start))

    plot_kvargs = {
        'price': raw_validation_price,
        'y_predicted': y_predicted_valid,
        'point': point,
        'win_size': win_size,
        'future': future
    }
    plot_3class_colored_prediction(**plot_kvargs)

    #close keras session
    K.clear_session()

    return history.history, metrics.get_scores(), plot_kvargs

#TODO
# add a lot of performance measures
# write a description MD file
# add Logger

# save model file in the same experiment folder - do it better!!



if __name__ == '__main__':
    logger.info("::: START :::")
    logger.info("   TensorFlow = " + tf.__version__)
    logger.info("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    # we can add a single experiment to artemis by single_train
    # or run add_all_experiments_variants() to add a numebr of experiments all together

    variant_test = rnn_1_train_basic.add_variant('test', lr=0.001, batch_size=1024, epochs=1)
    variant_small = rnn_1_train_basic.add_variant('small', lr=0.0005, batch_size=6000,epochs = 30)
    variant_medium = rnn_1_train_basic.add_variant('medium', train_coin_list=TRAIN_COINS_LIST, lr=0.001, batch_size=6000, epochs = 30)

    record_test = variant_test.run(keep_record=True)
    shutil.move("models/lstm_model.h5", record_test.get_dir())

    record_medium = variant_medium.run(keep_record=True)
    shutil.move("models/lstm_model.h5", record_medium.get_dir())


    # variant = rnn_1_train_basic.get_variant('test')
    # records = variant.get_records()
    # records[0].info
    # f_to = records[0].get_dir()







# TODO:
# - leave what I have as price_direction_indicator, with two classes and make them balanced (undersampling? adjusting delta?)
#    - so we can build a signal as indicator changed from - to +
# - add one more ANN which is a signal by itself, with anomaly detection approach with NN

# - try to learn for market manipulation detection.. BTC is clearly manipulated, try to detect this event (Orders? )

# - try to add ETh to BTC as a predictor... may be it works as a predictor



