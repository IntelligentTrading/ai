__author__ = 'Alex Y'

from train import rnn_1_train_basic

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    logger.info("::: START :::")

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