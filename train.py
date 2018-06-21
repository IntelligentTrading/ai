import tensorflow as tf
import keras
from models.keras_models import build_lstm_model, predict_point_by_point
from artemis.experiments import ExperimentFunction
from data.data_sources import get_combined_cleaned_df
from data.datasets import one_coin_array_from_df, get_dataset_fused
from utils.plotting import plot_model_metrics, plot_3class_colored_prediction


#@ExperimentFunction
def single_run(TRAIN_COINS_LIST, res_period, win_size, future, return_target, label_func, data_dim, num_classes, lstm_layers, lr, batch_size, epochs):
    # build a dataset for training
    X_train, Y_train = get_dataset_fused(
        TRAIN_COINS_LIST,
        'prodcopy',
        res_period=res_period,
        win_size=win_size,
        future=future,
        return_target=return_target,
        label_func=label_func,
        num_classes=num_classes
    )

    # set a validation ts (BTC/2 here, can be changed)
    valid_data_df = get_combined_cleaned_df(transaction_coin='BTC', counter_coin=2, res_period=res_period)
    valid_price_array = valid_data_df['price'].values

    # build a model
    model = build_lstm_model(win_size, data_dim, num_classes, lstm_layers, lr)

    # train the model
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.15)

    plot_model_metrics(history)

    model.save('lstm_model.h5')

    ### plot colored prediction on train data
    # get
    y_predicted_train = predict_point_by_point(model, X_train)
    plot_3class_colored_prediction(price, y_predicted, point, win_size, future)



#TODO
# add 2 label classification
# add artemis
# add a lot of performance measures


if __name__ == '__main__':
    print("   TensorFlow = " + tf.__version__)
    print("   Keras = " + keras.__version__)

    # train ANN for short period (future= 3 periods = 6 h), 288=48h back
    # TODO: try to balance classes
    # TODO: run for different return targets and take the best performer

    ####################### parameters of the dataset and model
    # list all coin pairs for the training set
    TRAIN_COINS_LIST = [('BTC', 2)]

    res_period = '10min'
    win_size = 288  # 48h back
    future = 24  # 4h forward
    return_target = 0.007
    data_dim = 4

    label_func = 'label_3class_return_target'
    num_classes = 3

    lstm_layers = [
        {'layer':'input', 'units':50, 'dropout':0.15},
        {'layer':'l2',    'units':25, 'dropout':0.15},
        #{'layer':'l3',    'units':32, 'dropout':0.15},
        {'layer':'last',  'units':16, 'dropout':0.15}
    ]
    lr = 0.0005
    batch_size = 512  # might be up to 7000 if enough memory and GPU
    epochs = 3
    ###############################


    single_run(
        TRAIN_COINS_LIST,
        res_period,
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




# TODO:
# - leave what I have as price_direction_indicator, with two classes and make them balanced (undersampling? adjusting delta?)
#    - so we can build a signal as indicator changed from - to +
# - add one more ANN which is a signal by itself, with anomaly detection approach with NN

# - try to learn for market manipulation detection.. BTC is clearly manipulated, try to detect this event (Orders? )




