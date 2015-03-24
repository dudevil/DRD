__author__ = 'dudevil'

import pickle
import numpy as np
from sklearn.metrics import confusion_matrix


def normalize(imgs, std_reg=1e-5):
    """
    Performs per-pixel zero-mean unit varience standartization of images
    :param imgs:
    :param std_reg:
    :return:
    """
    return (imgs - imgs.mean(axis=0, keepdims=True)) / (imgs.std(axis=0, keepdims=True) + std_reg)


def save_network(layer_list, filename='data/tidy/net.pickle'):
    with open(filename, 'wb') as f:
        pickle.dump(layer_list, f, -1)


def load_network(filename='data/tidy/net.pickle'):
    with open(filename, 'r') as f:
        net = pickle.load(f)
    return net, net[-1]


def kappa(y_true, y_pred):
    """
    Quadratic kappa score: http://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation
    Implementaion mostly taken from: http://scikit-learn-laboratory.readthedocs.org/en/latest/_modules/skll/metrics.html

    :param y_true:
    :param y_pred:
    :return:
    """
    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        print("Kappa values must be integers or strings")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    weights = np.empty((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = abs(i - j) ** 2

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k

