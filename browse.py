__author__ = 'Alex Y'

from train import rnn_1_train_basic

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TRAIN_COINS_LIST = [
    ('ETH', 2), ('XRP',2), ('ETC',2), ('DASH',2), ('LTC',2),
    ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0)
]

if __name__ == '__main__':
    logger.info("::: START :::")

    variant_test = rnn_1_train_basic.add_variant('test', lr=0.001, batch_size=1024, epochs=1)
    variant_small = rnn_1_train_basic.add_variant('small', lr=0.0005, batch_size=6000, epochs=30)
    variant_medium = rnn_1_train_basic.add_variant('medium', train_coin_list=TRAIN_COINS_LIST, lr=0.0008,
                                                   batch_size=6500, epochs=100)

    rnn_1_train_basic.browse()



    # Browse Commands:
    #   run all
    #   compare all
    #   display 1
    # show 0     show the results
    #   view full           View all columns (the default view)
    # delete 4.1          Delete record 1 of experiment 4
    # delete unfinished   Delete all experiment records that have not run to completion
    # delete 4-6          Delete all records from experiments 4, 5, 6.  You will be asked to confirm the deletion.
    ###########################################################