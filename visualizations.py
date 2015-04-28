__author__ = 'dudevil'

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import Greys_r
import numpy as np

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