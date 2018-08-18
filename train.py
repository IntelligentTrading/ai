__author__ = 'AlexBioY'

import shutil
import tensorflow as tf
import keras

from src.data.settings import TRAIN_COINS_LIST_BASIC, TRAIN_COINS_LIST_TOP20
from src.models.keras_models import rnn_train_basic

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



if __name__ == '__main__':
    logger.info(">>>>>>>>>>>   START <<<<<<<<<<<< ")
    logger.info("   TensorFlow = " + tf.__version__)
    logger.info("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    ################## local test variant ##########
    test_transform = 'short_60m_96_4_3class_return_0.02'
    test_layers = [
        {'layer': 'input', 'units': 128, 'dropout': 0.1},
        {'layer': 'l2', 'units': 64, 'dropout': 0.05},
        {'layer': 'l3', 'units': 32, 'dropout': 0.03},
        {'layer': 'l4', 'units': 32, 'dropout': 0.02},
        {'layer': 'last', 'units': 16, 'dropout': 0.001}
    ]
    variant_test = rnn_train_basic.add_variant(
        variant_name='test',
        ds_transform=test_transform,
        lstm_layers=test_layers,
        train_coin_list = TRAIN_COINS_LIST_BASIC,
        lr=0.009,
        batch_size=2048,
        epochs=120
    )


    ################## old variant ####################
    ds_transform_1 = 'basic_10m_288_24_3class_return0.01'
    lstm_layers_1 = [
        {'layer':'input', 'units':90, 'dropout':0.15},
        {'layer':'l2',    'units':64, 'dropout':0.15},
        {'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.1}
    ]
    variant_medium = rnn_train_basic.add_variant(
        variant_name=ds_transform_1,  # comes from a decorator
        ds_transform=ds_transform_1,
        lstm_layers=lstm_layers_1,
        train_coin_list=TRAIN_COINS_LIST_BASIC,
        lr=0.002,
        batch_size=6500,
        epochs = 200
    )

    ################## SHORT variant  ###################
    ##  another transformation, less dropout
    ds_transform_2 = 'short_60m_96_4_3class_return_0.02'
    lstm_layers_2 = [
        {'layer': 'input', 'units': 96, 'dropout': 0.1},
        {'layer': 'l2', 'units': 64, 'dropout': 0.1},
        {'layer': 'l3', 'units': 32, 'dropout': 0.05},
        {'layer': 'l4', 'units': 20, 'dropout': 0.05},
        {'layer': 'last', 'units': 16, 'dropout': 0.01}
    ]
    variant_short = rnn_train_basic.add_variant(
        variant_name=ds_transform_2,  # comes from a decorator
        ds_transform=ds_transform_2,
        lstm_layers=lstm_layers_2,
        train_coin_list=TRAIN_COINS_LIST_TOP20,
        lr=0.009,
        batch_size=6000,
        epochs=50
    )
    ##################################################


    ############### RUN variants ######################
    record_test = variant_test.run(keep_record=True, display_results=True)
    shutil.move("models/lstm_" + test_transform + ".h5", record_test.get_dir())

    # move generated model to the same artemis folder where all information is placed
    #record_medium = variant_medium.run(keep_record=True)
    #shutil.move("models/lstm_" + ds_transform_1 + ".h5", record_medium.get_dir())


    #record_short = variant_short.run(keep_record=True, display_results=True)
    #shutil.move("models/lstm_" + ds_transform_2 + ".h5", record_short.get_dir())



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




