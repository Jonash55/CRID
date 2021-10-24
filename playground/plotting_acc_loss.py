""" Module for plotting accuracy and loss vs no. of epochs after training """

import matplotlib.pyplot as plt


def plot_accuracies(history_of_training: list) -> None:
    """
    Plots accuracy vs no. of epochs

    Parameters
    ----------
    history_of_training: list
        accuracy values as floats to plot

    Returns
    -------
    """

    accuracies = [x['val_acc'] for x in history_of_training]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')


def plot_losses(history_of_training: list) -> None:
    """
    Plots loss vs no. of epochs for training and validation datasets

    Parameters
    ----------
    history_of_training: list
        loss values as floats to plot

    Returns
    -------
    """

    train_losses = [x.get('train_loss') for x in history_of_training]
    val_losses = [x['val_loss'] for x in history_of_training]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
