from ittai.data.data_sources import ittconnection
from ittai.data.datasets import ts_to_dataset_classification
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
    db_connection = ittconnection('prodcopy')

    ts_to_dataset_classification(data_df, win_size, stride, future, delta)


    # demo_drunkards_walk.browse()
    # Try
    #   run all
    #   compare all
    #   display 1
    pass



