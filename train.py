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

    ################## local variant ##################
    local_short_transform = 'short_60m_48_4_3class_return_0.02'
    local_short_layers = [
        {'layer': 'input', 'units': 128, 'dropout': 0.1},
        {'layer': 'l2', 'units': 64, 'dropout': 0.05},
        {'layer': 'l3', 'units': 32, 'dropout': 0.03},
        {'layer': 'l4', 'units': 32, 'dropout': 0.02},
        {'layer': 'last', 'units': 16, 'dropout': 0.001}
    ]
    variant_local_short = rnn_train_basic.add_variant(
        variant_name='local_short',
        ds_transform=local_short_transform,
        lstm_layers=local_short_layers,
        train_coin_list = TRAIN_COINS_LIST_BASIC,
        lr=0.01,
        batch_size=4096,
        epochs=6
    )


    ################## SHORT server variant  ###################
    ##  another transformation, less dropout
    ds_transform_server_short = 'short_60m_48_4_3class_return_0.02'
    lstm_layers_server_short = [
        {'layer': 'input', 'units': 128, 'dropout': 0.15},
        {'layer': 'l2', 'units': 64, 'dropout': 0.1},
        {'layer': 'l3', 'units': 32, 'dropout': 0.1},
        {'layer': 'l4', 'units': 32, 'dropout': 0.05},
        {'layer': 'last', 'units': 16, 'dropout': 0.01}
    ]
    variant_short = rnn_train_basic.add_variant(
        variant_name='server_short',  # comes from a decorator
        ds_transform=ds_transform_server_short,
        lstm_layers=lstm_layers_server_short,
        train_coin_list=TRAIN_COINS_LIST_TOP20,
        lr=0.02,
        batch_size=6000,
        epochs=200
    )
    ##################################################


    ############### RUN variants ######################
    # record_test = variant_local_short.run(keep_record=True, display_results=True)
    # shutil.move("models/lstm_" + local_short_transform + ".h5", record_test.get_dir())


    record_server_short = variant_short.run(keep_record=True)
    shutil.move("models/lstm_" + ds_transform_server_short + ".h5", record_server_short.get_dir())


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




