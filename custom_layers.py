__author__ = 'dudevil'

from lasagne import layers
import theano.tensor as T
import numpy as np


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
