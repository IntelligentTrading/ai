__author__ = 'AlexBioY'
import matplotlib.pyplot as plt
import numpy as np
from src.data.settings import DATASET_TRANSFORM

np.set_printoptions(precision=3)


def plot_class_colored_prediction(model_summary_str, final_val_scores, price, y_predicted, point, win_size, future, y_true, ds_transform): # , y_true
    start_of_train_position = point
    position_on_plot = point + win_size
    end_of_future_position = point + win_size + future

    num_classes = DATASET_TRANSFORM[ds_transform].num_classes


    # color each dot according to prediction: if UP-> green, if DOWN->red
    # for PREDICTED
    # skip first win_size prices, since we dont have prediction for them

    col3 = []
    col3_true = []
    # skip first win_size prices, since we dont have prediction for them
    for i in range(0,win_size):
        col3.append('yellow')
        col3_true.append('yellow')

    # set different colors for different label functions
    color_label_1 = 'green'
    color_label_2 = 'red'
    if DATASET_TRANSFORM[ds_transform].label_func == 'label_3class_max_hit':
        color_label_1 = 'green'
        color_label_2 = 'lime'

    # now color according to prediction
    for p in y_predicted:
        idx = np.argmax(p) - (num_classes-3)
        if idx == 1:
            color = color_label_1
        elif idx == 2:
            color = color_label_2
        else:
            color = 'black'
        col3.append(color)


    for p in y_true:
        idx = np.argmax(p) - (num_classes-3)
        if idx == 1:
            color = color_label_1
        elif idx == 2:
            color = color_label_2
        else:
            color = 'black'
        col3_true.append(color)

    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

    plt.figtext(0.6, 0.65, "DATA: " + ds_transform + "\n" + model_summary_str )

    ax2.scatter(range(price.shape[0]), price, c=col3_true, s=7)
    ax2.set_title(" TRUE LABELS")

    ax1.set_title(" Predicted: [SAME, UP, DOWN]::   F1: %s,  PRECISION: %s,  RECALL: %s" % (str(final_val_scores['f1']), str(final_val_scores['precision']), str(final_val_scores['recall'])))
    ax1.scatter(range(price.shape[0]), price, c=col3, s=7)

    ax1.axvline(start_of_train_position, color='blue')
    ax1.axvline(position_on_plot, color=col3[position_on_plot], lw=1)

    ax1.axvline(end_of_future_position, color=col3[position_on_plot], lw=1)


    plt.show(block=True)


def plot_model_results(results):
    # results is a dictionary of dictionaries of all returning results from the experiment

    history = results[0]
    train_val_scores = results[1]
    plot_kvargs = results[2]
    model_config_dict = results[3]
    final_val_scores = results[4]

    print("===== Data Transformation ======")
    print(plot_kvargs['ds_transform'])

    print("===== Model summary:")
    model_summary_str = "MODEL: \n"
    for layer in model_config_dict:
        model_summary_str = model_summary_str  + \
                            str(layer['class_name']) + ' >> units: ' + str(layer['config']['units']) + \
                            ',  activation: ' +  str(layer['config']['activation']) + \
                            ',  dropout: ' +  str(layer['config']['dropout'] if 'dropout' in layer['config'] else 'NA') + "\n"

    print(model_summary_str)

    print("======= Training progress of loss and accuracy (based on keras):")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9, 5))
    axes[0, 0].set_title('train loss')
    axes[0, 0].plot(history['loss'])

    axes[0, 1].set_title('validation loss')
    axes[0, 1].plot(history['val_loss'], c='orange')

    axes[1, 0].set_title('Train Accuracy')
    axes[1, 0].plot(history['acc'])

    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].plot(history['val_acc'], c='orange')

    axes[0, 2].set_title('BTC Validation Precision')
    axes[0, 2].plot(train_val_scores['precision'])

    axes[1, 2].set_title('BTC Validation Recall')
    axes[1, 2].plot(train_val_scores['recall'])

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

    plot_class_colored_prediction(model_summary_str, final_val_scores, **plot_kvargs)
