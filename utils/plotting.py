import matplotlib.pyplot as plt
import numpy as np

def plot_3class_colored_prediction(price, y_predicted, point, win_size, future):
    start_of_train_position = point
    position_on_plot = point + win_size
    end_of_future_position = point + win_size + future

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))
    col3 = []
    for p in y_predicted:
        idx = np.argmax(p)
        if idx == 1:
            color = 'green'
        elif idx == 2:
            color = 'red'
        else:
            color = 'black'
        col3.append(color)

    ax1.scatter(range(price.shape[0]), price, c=col3, s=4)

    ax1.axvline(position_on_plot, color=col3[position_on_plot], lw=2)
    ax1.axvline(end_of_future_position, color=col3[position_on_plot], lw=2)
    ax1.axvline(start_of_train_position, color='blue')

    plt.show(block=False)


def plot_model_metrics(history):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
    axes[0, 0].set_title('train loss')
    axes[0, 0].plot(history.history['loss'])

    axes[0, 1].set_title('validation loss')
    axes[0, 1].plot(history.history['val_loss'], c='orange')

    axes[1, 0].set_title('Train Accuracy')
    axes[1, 0].plot(history.history['acc'])

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].plot(history.history['val_acc'], c='orange')

    plt.show(block=False)
    print(history.history.keys())