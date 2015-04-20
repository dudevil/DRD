__author__ = 'dudevil'

import pickle
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
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


def images_byerror(y_pred, y_true, images):
    diff = np.abs(y_true - y_pred)
    order = diff.argsort()[::-1]
    return pd.Series(index=images[order], data=diff[order])


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


def gaussian_filter(kernel_shape):
    x = np.zeros((kernel_shape, kernel_shape), dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma**2
        return  1./Z * np.exp(-(x**2 + y**2) / (2. * sigma**2))

    for i in xrange(kernel_shape):
        for j in xrange(kernel_shape):
            x[i,j] = gauss(i-4., j-4.)

    return x / np.sum(x)


def lecun_lcn(input, img_shape, kernel_shape, threshold=1e-4):
    """
    Yann LeCun's local contrast normalization
    This is performed per-colorchannel!!!

    http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf
    """
    input = input.reshape((input.shape[0], 1, input.shape[1], input.shape[2]))
    X = T.matrix(dtype=input.dtype)
    X = X.reshape((len(input), 1, img_shape[0], img_shape[1]))

    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = theano.shared(gaussian_filter(kernel_shape).reshape(filter_shape))

    convout = conv.conv2d(input=X,
                          filters=filters,
                          image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                          filter_shape=filter_shape,
                          border_mode='full')

    # For each pixel, remove mean of 9x9 neighborhood
    mid = int(np.floor(kernel_shape / 2.))
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]

    # Scale down norm of 9x9 patch if norm is bigger than 1
    sum_sqr_XX = conv.conv2d(input= centered_X**2,
                             filters=filters,
                             image_shape=(input.shape[0], 1, img_shape[0], img_shape[1]),
                             filter_shape=filter_shape,
                             border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = T.mean(T.flatten(denom, outdim=3), axis=2) #denom.mean(axis=[1, 2])
    divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor
    #new_X = theano.tensor.flatten(new_X, outdim=3)

    f = theano.function([X], new_X)
    return f(input)


def lcn_image(images, kernel_size=9):
    """
    This assumes image is 01c and the output will be c01 (compatible with conv2d)

    :param image:
    :param inplace:
    :return:
    """
    image_shape = (images.shape[1], images.shape[2])
    if len(images.shape) == 3:
        # this is greyscale images
        output = lecun_lcn(images, image_shape, kernel_size)
    else:
        # color image, assume RGB
        r = images[:, :, :, 0]
        g = images[:, :, :, 1]
        b = images[:, :, :, 2]

        output = np.concatenate((
            lecun_lcn(r, image_shape, kernel_size),
            lecun_lcn(g, image_shape, kernel_size),
            lecun_lcn(b, image_shape, kernel_size)),
            axis=1
        )
    return output


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """ Code adopted from here: https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/expr/preprocessing.py
        but can work with b01c and bc01 orderings

        An Analysis of Single-Layer
       Networks in Unsupervised Feature Learning". AISTATS 14, 2011.
       http://www.stanford.edu/~acoates/papers/coatesleeng_aistats_2011.pdf
    """
    assert X.ndim > 2, "X.ndim must be more than 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    aggr_axis = tuple(np.arange(len(X.shape) - 1) + 1)
    mean = np.mean(X, axis=aggr_axis, keepdims=True)
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy.
    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + np.var(X, axis=aggr_axis, ddof=ddof, keepdims=True)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + np.sum((X ** 2), axis=aggr_axis, keepdims=True)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.
    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X