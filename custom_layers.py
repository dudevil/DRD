__author__ = 'dudevil'

import theano
import theano.tensor as T
import numpy as np
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Can't import this from lasagne.pool
def pool_output_length(input_length, ds, st, ignore_border=True, pad=0):

    if input_length is None or ds is None:
        return None

    if ignore_border:
        output_length = input_length + 2 * pad - ds + 1
        output_length = (output_length + st - 1) // st

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if st >= ds:
            output_length = (input_length + st - 1) // st
        else:
            output_length = max(
                0, (input_length - ds + st - 1) // st) + 1

    return output_length

class SliceRotateLayer(layers.Layer):

    def __init__(self, incoming, name=None, patch_shape=(64, 64)):
        super(SliceRotateLayer, self).__init__(incoming, name)
        self.slice_shape = patch_shape

    def get_output_shape_for(self, input_shape):
        return input_shape[0] * 4, input_shape[1], self.slice_shape[0], self.slice_shape[1]

    def get_output_for(self, input, **kwargs):
        px, py = self.slice_shape # shortcut
        part0 = input[:, :, :px, :py] # 0 degrees
        part1 = input[:, :, :px, :-py-1:-1].dimshuffle(0, 1, 3, 2) # 90 degrees
        part2 = input[:, :, :-px-1:-1, :-py-1:-1] # 180 degrees
        part3 = input[:, :, :-px-1:-1, :py].dimshuffle(0, 1, 3, 2) # 270 degrees

        return T.concatenate([part0, part1, part2, part3], axis=0)


class RotateMergeLayer(layers.Layer):

    def get_output_shape_for(self, input_shape):
        return input_shape[0] / 4, np.prod(input_shape[1:]) * 4

    def get_output_for(self, input, **kwargs):
        input_r = input.reshape((4, self.input_shape[0] // 4, int(np.prod(self.input_shape[1:])))) # split out the 4* dimension
        return input_r.transpose(1, 0, 2).reshape(self.get_output_shape())


class StochasticPoolLayer(layers.Layer):

    def __init__(self, incoming, ds, strides=None, ignore_border=False, pad=(0, 0), random_state=42, **kwargs):
        super(StochasticPoolLayer, self).__init__(incoming, **kwargs)
        self.ds = ds
        self.ignore_border = ignore_border
        self.pad = pad
        self.st = ds if strides is None else strides
        if hasattr(random_state, 'multinomial'):
            self.rng = random_state
        else:
            self.rng = RandomStreams(seed=random_state)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             ds=self.ds[0],
                                             st=self.st[0],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[0],
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             ds=self.ds[1],
                                             st=self.st[1],
                                             ignore_border=self.ignore_border,
                                             pad=self.pad[1],
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, deterministic=False, **kwargs):
        # inspired by:
        # https://github.com/lisa-lab/pylearn2/blob/14b2f8bebce7cc938cfa93e640008128e05945c1/pylearn2/expr/stochastic_pool.py#L23
        batch, channels, nr, nc = self.input_shape
        pr, pc = self.ds
        sr, sc = self.st
        output_shape = self.get_output_shape()
        out_r, out_c = output_shape[2:]
        # calculate shape needed for padding
        pad_shape = list(output_shape)
        pad_shape[2] = (pad_shape[2] - 1) * sr + pr
        pad_shape[3] = (pad_shape[3] - 1) * sc + pc
        # allocate a new input tensor
        padded = T.alloc(0.0, *pad_shape)
        # get padding offset
        offset_x = (pad_shape[2] - nr) // 2
        offset_y = (pad_shape[3] - nc) // 2

        padded = T.set_subtensor(padded[:, :, offset_x:(offset_x + nr), offset_y:(offset_y + nc)], input)
        window = T.alloc(0.0, batch, channels, out_r, out_c, pr, pc)
        for row_within_pool in xrange(pr):
            row_stop = (output_shape[2] - 1) * sr + row_within_pool + 1
            for col_within_pool in xrange(pc):
                col_stop = (output_shape[3] - 1) * sc + col_within_pool + 1
                # theano dark magic
                win_cell = padded[:, :, row_within_pool:row_stop:sr, col_within_pool:col_stop:sc]
                window = T.set_subtensor(window[:, :, :, :, row_within_pool, col_within_pool], win_cell)
        # sum across pooling regions
        norm = window.sum(axis=[4, 5])
        # avoid zero-division
        norm = T.switch(T.eq(norm, 0.0), 1.0, norm)
        # normalize
        norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')

        if deterministic:
            res = (window * norm).sum(axis=[4, 5])
        else:
            prob = self.rng.multinomial(pvals=norm.reshape((batch * channels * out_r * out_c, pr * pc)),
                                        dtype=theano.config.floatX)
            # double max because of grad problems
            res = (window * prob.reshape((batch, channels, out_r, out_c,  pr, pc))).max(axis=5).max(axis=4)

        return T.cast(res, theano.config.floatX)



