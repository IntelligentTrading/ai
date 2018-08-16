__author__ = 'AlexBioY'

import shutil
import time
from artemis.experiments import ExperimentFunction
import tensorflow as tf
import keras
from keras import backend as K
from src.vizualization.plotting import plot_model_results, plot_3class_colored_prediction
from src.data.data_sources import get_combined_cleaned_onecoin_df
from src.data.datasets import get_dataset_manycoins_fused
from src.models.keras_models import build_lstm_model, Metrics
from src.data.settings import TRAIN_COINS_LIST_BASIC, TRAIN_COINS_LIST_TOP20, DATASET_TRANSFORM

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compare_trainings(dict_of_histories):
    print("you can add a training comparison here to show it in UI")


def display_train_result(results):
    plot_model_results(results)


#TODO: remove a lot of parameters, use DATASET_TRANSFORM, pass only name 'basic'
@ExperimentFunction(display_function=display_train_result,  is_root=True)
def rnn_1_train_basic(
        train_coin_list=[('BTC', 2), ('ETH', 0)],
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
        validation_split=0.15,
        callbacks=[metrics],
        verbose = 2         # 0 = silent, 1 = progress bar, 2 = one line per epoch
    )

    model.save("models/lstm_" + ds_transform + ".h5")


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
    X_valid, Y_valid = get_dataset_manycoins_fused(COINS_LIST=[(VALID_COIN, VALID_COUNTER)], db_name='prodcopy', ds_transform=ds_transform)

    ### plot colored prediction on train data
    # get
    point=2500
    logger.info(">>>>>>>>>>  PREDICTING and PLOTTING on validation dataset (BTC)")
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

    return history.history, metrics.get_scores(), plot_kvargs, model.summary()



if __name__ == '__main__':
    logger.info(">>>>>>>>>>>   START <<<<<<<<<<<< ")
    logger.info("   TensorFlow = " + tf.__version__)
    logger.info("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    #variant_test = rnn_1_train_basic.add_variant('test', lr=0.001, batch_size=1024, epochs=1)
    #variant_small = rnn_1_train_basic.add_variant('small', lr=0.0005, batch_size=6000,epochs = 30)

    ################## old variant ####################
    ds_transform_1 = 'basic_10m_288_24_3class_return0.01'
    lstm_layers_1 = [
        {'layer':'input', 'units':90, 'dropout':0.15},
        {'layer':'l2',    'units':64, 'dropout':0.15},
        {'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.1}
    ]
    variant_medium = rnn_1_train_basic.add_variant(
        'medium',
        ds_transform=ds_transform_1,
        lstm_layers=lstm_layers_1,
        train_coin_list=TRAIN_COINS_LIST_BASIC,
        lr=0.002,
        batch_size=6500,
        epochs = 200
    )

    ################## SHORT variant  ###################
    ##  another transformation, less dropout
    ds_transform_2 = 'short_60m_96_4_3class_return_0.05'
    lstm_layers_2 = [
        {'layer': 'input', 'units': 96, 'dropout': 0.1},
        {'layer': 'l2', 'units': 64, 'dropout': 0.1},
        {'layer': 'l3', 'units': 32, 'dropout': 0.05},
        {'layer': 'l4', 'units': 20, 'dropout': 0.05},
        {'layer': 'last', 'units': 16, 'dropout': 0.01}
    ]
    variant_short = rnn_1_train_basic.add_variant(
        variant_name=ds_transform_2,  # comes from a decorator
        ds_transform=ds_transform_2,
        lstm_layers=lstm_layers_2,
        train_coin_list=TRAIN_COINS_LIST_BASIC,
        lr=0.002,
        batch_size=6000,
        epochs=50
    )
    ##################################################


    ############### RUN variants ######################
    #record_test = variant_test.run(keep_record=True)
    #shutil.move("models/lstm_model.h5", record_test.get_dir())

    # move generated model to the same artemis folder where all information is placed
    record_medium = variant_medium.run(keep_record=True)
    shutil.move("models/lstm_" + ds_transform_1 + ".h5", record_medium.get_dir())


    record_short = variant_short.run(keep_record=True)
    shutil.move("models/lstm_" + ds_transform_2 + ".h5", record_short.get_dir())



    logger.info(">>>>>>>>>>>>>> ::: COMPLETED ::: <<<<<<<<<<<<<< ")


    # variant = rnn_1_train_basic.get_variant('test')
    # records = variant.get_records()
    # records[0].info
    # f_to = records[0].get_dir()







# TODO:
# - try validation on several currencies, not only BTC
# - leave what I have as price_direction_indicator, with two classes and make them balanced (undersampling? adjusting delta?)
#    - so we can build a signal as indicator changed from - to +
# - add one more ANN which is a signal by itself, with anomaly detection approach with NN

# - try to learn for market manipulation detection.. BTC is clearly manipulated, try to detect this event (Orders? )




