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
        ax.imshow(f[i][31], interpolation='none', cmap=cm)