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
    logger.info(">>>>>>>>>>>   START TRAINING SCRIPT  <<<<<<<<<<<< ")
    logger.info("   TensorFlow = " + tf.__version__)
    logger.info("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    ################## local variant ##################
    local_short_transform = 'short_60m_96_8_3class_return_0.04'
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
        batch_size=64,
        epochs=10
    )


    ################## SHORT server variant  ###################
    ##  another transformation, less dropout
    ds_transform_server_short = 'short_60m_160_8_3class_return_0.03'
    lstm_layers_server_short = [
        {'layer': 'input', 'units': 256, 'dropout': 0.01},
        {'layer': 'l2', 'units': 128, 'dropout': 0.01},
        {'layer': 'l3', 'units': 64, 'dropout': 0.01},
        {'layer': 'l4', 'units': 64, 'dropout': 0.01},
        {'layer': 'l5', 'units': 64, 'dropout': 0.01},
        {'layer': 'l6', 'units': 64, 'dropout': 0.01},
        {'layer': 'last', 'units': 32, 'dropout': 0.01}
    ]
    variant_short = rnn_train_basic.add_variant(
        variant_name='server_short',  # comes from a decorator
        ds_transform=ds_transform_server_short,
        lstm_layers=lstm_layers_server_short,
        lr=0.001,
        batch_size=512,
        epochs=3
    )
    ##################################################

    ################## MEDIUM server variant  ###################
    ##  another transformation, less dropout
    ds_transform_server_medium = 'medium_240m_100_20_3class_return_0.1'
    lstm_layers_server_medium = [
        {'layer': 'input', 'units': 128, 'dropout': 0.01},
        {'layer': 'l2', 'units': 256, 'dropout': 0.01},
        {'layer': 'l3', 'units': 256, 'dropout': 0.01},
        {'layer': 'l4', 'units': 128, 'dropout': 0.05},
        {'layer': 'l5', 'units': 64, 'dropout': 0.05},
        {'layer': 'l6', 'units': 64, 'dropout': 0.05},
        {'layer': 'last', 'units': 32, 'dropout': 0.01}
    ]
    variant_medium = rnn_train_basic.add_variant(
        variant_name='server_medium',  # comes from a decorator
        ds_transform=ds_transform_server_medium,
        lstm_layers=lstm_layers_server_medium,
        lr=0.001,
        batch_size=512,
        epochs=10
    )
    ##################################################

    ################## LONG server variant  ###################
    ds_transform_server_long = 'long_1440m_28_10_class3_return_0.1'
    lstm_layers_server_long = [
        {'layer': 'input', 'units': 128, 'dropout': 0.01},
        {'layer': 'l2', 'units': 256, 'dropout': 0.1},
        {'layer': 'l3', 'units': 256, 'dropout': 0.01},
        {'layer': 'l4', 'units': 128, 'dropout': 0.01},
        {'layer': 'last', 'units': 64, 'dropout': 0.01}
    ]

    variant_long = rnn_train_basic.add_variant(
        variant_name='server_long',  # comes from a decorator
        ds_transform=ds_transform_server_long,
        lstm_layers=lstm_layers_server_long,
        lr=0.0008,
        batch_size=7000,
        epochs=5
    )
    ##################################################


    ################## 2 classЖ LONG server variant -  ###################
    ds_transform_server_long_2cl = 'long_1440m_28_4_class2'
    lstm_layers_server_long = [
        {'layer': 'input', 'units': 128, 'dropout': 0.01},
        {'layer': 'l2', 'units': 256, 'dropout': 0.01},
        {'layer': 'l3', 'units': 256, 'dropout': 0.01},
        {'layer': 'l4', 'units': 128, 'dropout': 0.01},
        {'layer': 'last', 'units': 64, 'dropout': 0.01}
    ]

    variant_long_2class = rnn_train_basic.add_variant(
        variant_name='server_long_2class',  # comes from a decorator
        ds_transform=ds_transform_server_long_2cl,
        lstm_layers=lstm_layers_server_long,
        lr=0.0008,
        batch_size=7000,
        epochs=9
    )
    ##################################################


    ############### RUN variants ######################
    # record_test = variant_local_short.run(keep_record=True, display_results=True)
    # shutil.move("models/lstm_" + local_short_transform + ".h5", record_test.get_dir())
    #


    # logger.info('================ start long - 2 class training  ===============')
    # record_server_long_2cl = variant_long_2class.run(keep_record=True)
    # shutil.move("models/lstm_" + ds_transform_server_long_2cl + ".h5", record_server_long_2cl.get_dir())

    # logger.info('================ start long training  ===============')
    # record_server_long = variant_long.run(keep_record=True)
    # shutil.move("models/lstm_" + ds_transform_server_long + ".h5", record_server_long.get_dir())


    # logger.info('================ start medium training  ===============')
    # record_server_medium = variant_medium.run(keep_record=True)
    # shutil.move("models/lstm_" + ds_transform_server_medium + ".h5", record_server_medium.get_dir())
    #
    #
    #
    logger.info('================ start short training  ===============')
    record_server_short = variant_short.run(keep_record=True)
    shutil.move("models/lstm_" + ds_transform_server_short + ".h5", record_server_short.get_dir())





    # TODO run same predictions but based on one hour resample, i.e. 60min


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




