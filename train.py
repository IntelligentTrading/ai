
from data.datasets import ts_to_dataset_onecoin, get_dataset_fused
from models.keras_models import lstm_model
from artemis.experiments import ExperimentFunction


@ExperimentFunction
def train_model(model, epochs, X_train, Y_train):


    #epochs = 40  # 100
    #model = build_model()

    history = model.fit(
        X_train,
        Y_train,
        batch_size=7000,
        epochs=epochs,
        validation_split=0.15)


if __name__ == '__main__':

    # list all coin pairs for the training set
    TRAIN_COINS_LIST = [('BTC', 2), ('ETH', 2), ('ETH', 0), ("ETC", 0), ('OMG', 0), ('XRP', 0), ('XMR', 0), ('LTC', 0)]

    # convert
    X_train, Y_train = get_dataset_fused(TRAIN_COINS_LIST, 'prodcopy', res_period = '10min', win_size = 200, future = 90, return_target = 0.02)
    model = lstm_model()
    epochs=40
    train_model(model, epochs, X_train, Y_train)


    # demo_drunkards_walk.browse()
    # Try
    #   run all
    #   compare all
    #   display 1
    pass



