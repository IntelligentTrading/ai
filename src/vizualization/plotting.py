__author__ = 'AlexBioY'
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)


def plot_3class_colored_prediction(price, y_predicted, point, win_size, future, y_true): # , y_true
    start_of_train_position = point
    position_on_plot = point + win_size
    end_of_future_position = point + win_size + future

    # color each dor according to prediction: if UP-> green, if DOWN->red
    # for PREDICTED
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

    col3_true = []
    for p in y_true:
        idx = np.argmax(p)
        if idx == 1:
            color = 'green'
        elif idx == 2:
            color = 'red'
        else:
            color = 'black'
        col3_true.append(color)

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
    ax2.scatter(range(price.shape[0]), price, c=col3_true, s=1)
    ax2.set_title(" TRUE LABELS ")

    ax1.scatter(range(price.shape[0]), price, c=col3, s=1)
    ax1.set_title(" predicted Labels ")

    ax1.axvline(start_of_train_position, color='blue')
    ax1.axvline(position_on_plot, color=col3[position_on_plot], lw=1)

    ax1.axvline(end_of_future_position, color=col3[position_on_plot], lw=1)

    plt.show(block=False)


def plot_model_results(results):
    # results is a dictionary of dictionaries of all returning results from the experiment

    history = results[0]
    train_val_scores = results[1]
    plot_kvargs = results[2]
    model_config_dict = results[3]
    final_val_scores = results[4]

    print("===== Data Transformation ======")

    print("===== Model summary:")
    for layer in model_config_dict:
        print(
            str(layer['class_name']) + '= units:' + str(layer['config']['units']) +
              ', activation: ' + str(layer['config']['activation'] +
              ', dropout: '
         ))

    print("======= Training progress of loss and accuracy (based on keras):")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes[0, 0].set_title('train loss')
    axes[0, 0].plot(history['loss'])

    axes[0, 1].set_title('validation loss')
    axes[0, 1].plot(history['val_loss'], c='orange')

    axes[1, 0].set_title('Train Accuracy')
    axes[1, 0].plot(history['acc'])

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].plot(history['val_acc'], c='orange')

    plt.show(block=False) # block=False

    print(" Scores from Training validation set [SAME, UP, DOWN] :")
    print(" f1        :" + str(train_val_scores['f1'][-1]))
    print(" recall    :" + str(train_val_scores['recall'][-1]))
    print(" precision :" + str(train_val_scores['precision'][-1]))

    # plot colored price
    print("======= Plot prediction on BTC ==== ")

    print("  [same,up,down]>> PRECISION: tp/(tp+fp)  ||  RECALL: tp/(tp+fn)")
    print('  F1: %s ||  PRECISION: %s  ||  RECALL: %s' %
          (str(final_val_scores['f1']), str(final_val_scores['precision']), str(final_val_scores['recall'])))

    plot_3class_colored_prediction(**plot_kvargs)
