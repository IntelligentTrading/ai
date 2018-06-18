
from data.datasets import build_dataset_array_from_df, get_dataset_fused
from models.keras_models import build_lstm_model
from artemis.experiments import ExperimentFunction


@ExperimentFunction
def single_run(TRAIN_COINS_LIST, win_size, future, return_target, label_func, data_dim, num_classes, lstm_layers, lr, batch_size, epochs):

    # build a dataset for training
    X_train, Y_train = get_dataset_fused(
        TRAIN_COINS_LIST,
        'prodcopy',
        res_period='10min',
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func
    )
    print(X_train.shape)
    print(Y_train.shape)

    # build a model
    model = build_lstm_model(win_size, data_dim, num_classes, lstm_layers, lr)

    history = model.fit(
        X_train,
        Y_train,
        batch_size=7000,
        epochs=epochs,
        validation_split=0.15)

    model.save('lstm_model.h5')






if __name__ == '__main__':

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    ####################### parameters of the dataset and model
    # list all coin pairs for the training set
    TRAIN_COINS_LIST = [('BTC', 2), ('ETH', 2), ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0)]

    win_size = 288
    future = 18
    return_target = 0.02
    data_dim = 4

    label_func = label_3class_return_target()
    num_classes = 3

    lstm_layers = [
        {'layer':'input', 'units':100, 'dropout':0.15},
        {'layer':'l2',    'units':64, 'dropout':0.15},
        {'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.15}
    ]
    lr = 0.0005
    ###############################

    single_run(
        TRAIN_COINS_LIST,
        win_size,
        future,
        return_target,
        label_func,
        data_dim,
        num_classes,
        lstm_layers,
        lr,
        batch_size,
        epochs
    )

    # demo_drunkards_walk.browse()
    # Try
    #   run all
    #   compare all
    #   display 1




# TODO: try to learn for manipulation detection.. BTC is clearly manipulated, try to detect this event (Orders? )




