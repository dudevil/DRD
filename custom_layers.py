__author__ = 'dudevil'

import theano
import theano.tensor as T
import numpy as np
from lasagne import layers
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def leaky_relu(x, alpha=3.0):
    return T.maximum(x, x * (1.0 / alpha))


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

    def __init__(self, incoming, ds, strides=None, ignore_border=False, random_state=42, **kwargs):
        super(StochasticPoolLayer, self).__init__(incoming, **kwargs)
        self.ds = ds
        self.ignore_border = ignore_border
        self.strides = strides if strides is not None else ds
        if hasattr(random_state, 'multinomial'):
            self.rng = random_state
        else:
            self.rng = RandomStreams(seed=random_state)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        # this assumes bc01 input ordering
        output_shape[2] = (input_shape[2] - self.ds[0]) // self.strides[0] + 1
        output_shape[3] = (input_shape[3] - self.ds[1]) // self.strides[1] + 1
        return tuple(output_shape)

    def get_output_for(self, input, deterministic=False, **kwargs):
        # inspired by:
        # https://github.com/lisa-lab/pylearn2/blob/14b2f8bebce7cc938cfa93e640008128e05945c1/pylearn2/expr/stochastic_pool.py#L23
        batch, channels, nr, nc = self.input_shape
        pr, pc = self.ds
        sr, sc = self.strides

        out_r, out_c = self.get_output_shape_for(self.input_shape)[2:]

        window = T.alloc(0.0, batch, channels, out_r, out_c, pr, pc)
        for row_within_pool in xrange(out_r):
            for col_within_pool in xrange(out_c):
                win_cell = input[:, :, row_within_pool:out_r * sr:sr, col_within_pool:out_c * sc:sc]
                window = T.set_subtensor(window[:, :, :, :, row_within_pool, col_within_pool], win_cell)

        norm = window.sum(axis=[4, 5])
        norm = T.switch(T.eq(norm, 0.0), 1.0, norm)
        norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')

        if deterministic:
            res = (window * norm).sum(axis=[4, 5])
        else:
            prob = self.rng.multinomial(pvals=norm.reshape((batch * channels * out_r * out_c, pr * pc)),
                                        dtype='float32')
            res = (window * prob.reshape((batch, channels, out_r, out_c,  pr, pc))).max(axis=5).max(axis=4)

        return T.cast(res, theano.config.floatX)


class RandomizedReLu(layers.Layer):

    def __init__(self, input_layer, a_min=3., a_max=8., random_state=42):
        super(RandomizedReLu, self).__init__(input_layer)
        self.a_min = a_min
        self.a_max = a_max
        self.rng = RandomStreams(seed=random_state)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            res = T.maximum(input, 2 * input / (self.a_max - self.a_min))
        else:
            batch_size, channels, _, _ = self.get_output_shape()
            a = self.rng.uniform(size=(batch_size, channels), low=self.a_min, high=self.a_max)
            a = a.dimshuffle((0, 1, 'x', 'x'))
            res = T.maximum(input, input / a)
        return res

