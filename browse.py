__author__ = 'Alex Y'

from train import rnn_train_basic
from src.data.settings import TRAIN_COINS_LIST_BASIC, TRAIN_COINS_LIST_TOP20

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    logger.info("::: START :::")

    variant_small = rnn_train_basic.add_variant('small')
    variant_medium = rnn_train_basic.add_variant('medium')
    variant_test = rnn_train_basic.add_variant(variant_name='test')

    rnn_train_basic.browse()



    # Browse Commands:
    #   run all
    #   compare all
    #   display 1
    # show 0     show the results
    #   view full           View all columns (the default view)
    # delete 4.1          Delete record 1 of experiment 4
    # delete unfinished   Delete all experiment records that have not run to completion
    # delete 4-6          Delete all records from experiments 4, 5, 6.  You will be asked to confirm the deletion.

    # display 2.16

    ###########################################################