__author__ = 'dudevil'

import numpy as np
import theano
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import Greys_r
from utils import load_network

def show_weights(layer, bias=True, filt=-1, color=False):
    W = layer.W.get_value()
    b = layer.b.get_value()
    if not bias:
        b = np.zeros_like(b)
    f = [w + bb for w, bb in zip(W, b)]
    if not color:
        cm = Greys_r
    else:
        cm = None
    dim = int(np.ceil(np.sqrt(layer.num_filters)))
    fig = plt.figure()
    gs = gridspec.GridSpec(dim, dim, wspace=.1, hspace=0.1)
    for i in range(layer.num_filters):
        g = gs[i]
        ax = fig.add_subplot(g)
        ax.grid()
        ax.set_xticks([])
        ax.set_yticks([])
        if not filt:
            filt = np.random.randint(f[i].shape[0])
        ax.imshow(f[i].transpose(1, 2, 0), cmap=cm)

def pretty_print_confusion(conf_matrix, labels):
    labels = tuple(labels)
    print("   |    %d |    %d |    %d |    %d |    %d |" % labels)
    print("---|------|------|------|------|------|")
    for i, lab in enumerate(labels):
        print(" %d | %4d | %4d | %4d | %4d | %4d |" % ((lab, ) + tuple(conf_matrix[i])))


def plot_confusion_matrix(conf_matrix, labels=(0, 1, 2, 3, 4), normalize=True,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = conf_matrix.astype('float', copy=True) / conf_matrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = conf_matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(max(labels) + 1)
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    x_offset = -.25
    y_offset = .08
    for y in xrange(cm.shape[0]):
        for x in xrange(cm.shape[1]):
            plt.text(x + x_offset, y + y_offset, "%.2f" % cm[y,x])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def occlusion_heatmap(net, image, level=4, square_length=7):
    """
    Determines which parts of the image are important for the network classification
    Blackens square regions in the image and measures target class probability
    See this paper: http://arxiv.org/abs/1311.2901

    :param net: a path to a pickeled network or a (net, output) tuple
    :param image: an image to test, this function assumes the images has been preprocessed
        in the same manner as during net training, otherwise the results are unpredictable
    :param level: measure agains this class (currently only class 4 is supported)
    :param square_length: size of the square which is blacked out in the image

    :return: an array the same size as image with target class probabilities
    """
    if isinstance(net, basestring):
        net, output = load_network(net)
    else:
        net, output = net
    img = image.copy()
    shape = img.shape
    # assume the first layer is InputLayer
    net_shape = net[0].get_output_shape()
    batch_size = net_shape[0]
    assert shape == net_shape[1:]
    # allocate network input variable
    x_shared = theano.shared(np.zeros(net_shape, dtype=theano.config.floatX), borrow=True)
    # allocate array for the heatmap
    heat_array = np.zeros(shape[1:])
    pad = square_length // 2
    # this will contain images with occluded regions
    x_occluded = np.zeros((shape[1] * shape[2], net_shape[1], shape[1], shape[2]),
                          dtype=img.dtype)
    # occlude image regions
    for i, j in product(*map(range, shape[1:])):
        x_padded = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
        x_padded[:, i:i + square_length, j:j + square_length] = 0.
        x_occluded[i * shape[1] + j, ...] = x_padded[:, pad:-pad, pad:-pad]

    predict_proba = theano.function(
        [],
        output.get_output(x_shared, deterministic=True)[:, level - 1],
    )
    n_occluded = len(x_occluded)
    probas = np.zeros(n_occluded, dtype=theano.config.floatX)
    # get the probabilities for occluded images
    for b in xrange(n_occluded / batch_size):
        batch_slice = slice(b * batch_size, (b + 1) * batch_size)
        x_shared.set_value(x_occluded[batch_slice], borrow=True)
        probas[batch_slice] = predict_proba()
    #assign probabilities to heat_map
    for i, j in product(*map(range, shape[1:])):
        heat_array[i, j] = probas[i * shape[1] + j]

    return heat_array