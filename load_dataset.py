__author__ = 'dudevil'

import os.path
import pandas as pd
import numpy as np
import theano
from sklearn.cross_validation import StratifiedShuffleSplit
from skimage.io import imread


class DataLoader(object):

    def __init__(self,
                 datadir='data',
                 image_size=224,
                 chunk_size=1024,
                 random_state=0):
        self.datadir = datadir
        self.random_state = random_state
        self.image_size = image_size
        self.chunk_size = chunk_size

        labels = pd.read_csv(os.path.join(self.datadir, "trainLabels.csv"))
        # split the dataset to train and 10% validation (3456 is closest to 10% divisible by batch size 128)
        sss = StratifiedShuffleSplit(labels.level, 1, test_size=3456, random_state=random_state)
        self.train_index, self.valid_index = list(sss).pop()
        # get train and validation labels
        self.train_labels = labels.level[self.train_index]
        self.valid_labels = labels.level[self.valid_index]
        # prepare train and test image files
        self.train_images = labels.image[self.train_index].apply(lambda img:
                                                                 os.path.join("data", "train", "resized", img + ".png"))
        self.valid_images = labels.image[self.valid_index].apply(lambda img:
                                                                 os.path.join("data", "train", "resized", img + ".png"))
        self.test_images = [os.path.join(self.datadir, "test", "resized", img) for img in
                            os.listdir(os.path.join(self.datadir, "test", "resized"))]

    def valid_gen(self):
        # allocate an array for images
        images = np.zeros((len(self.valid_images), self.image_size, self.image_size, 3), dtype=np.float32)
        for i, image in enumerate(self.valid_images):
            images[i, ...] = imread(image)
        # return images and labels. We need to change the order of axis here, because
        # imread returns an array of shape (img_size, img_size, n_channels), while the conv layers
        # expect shape (batch_size, n_channels, img_size, img_size)
        return np.rollaxis(images, 3, 1), self.valid_labels.values.astype(np.int32)[np.newaxis].T

    def train_gen(self):
        # allocate an array for images
        images = np.zeros((self.chunk_size, self.image_size, self.image_size, 3), dtype=np.float32)
        n_chunks = int(np.ceil(len(self.train_images) * 1. / self.chunk_size))
        while n_chunks:
            for chunk in xrange(n_chunks):
                # prepare a slice of images to read during this pass
                chunk_end = (chunk + 1) * self.chunk_size
                # we need this to get images if the test set is not divisible by chunk_size

                if len(self.test_images) < chunk_end:
                    images = np.zeros((len(self.test_images) - chunk * self.chunk_size,
                                       self.image_size, self.image_size, 3))
                    chunk_end = len(self.test_images)
                chunk_slice = slice(chunk * self.chunk_size, chunk_end)
                # read a chunk of images
                for i, image in enumerate(self.train_images[chunk_slice]):
                    images[i, ...] = imread(image)
                # change axis order (see comments in valid_gen function) and yield images with labels
                yield np.rollaxis(images, 3, 1), self.train_labels[chunk_slice].values.astype(np.int32)[np.newaxis].T
        print("Number of images is less than chunk size.")

    def test_gen(self):
        images = np.zeros((self.chunk_size, self.image_size, self.image_size, 3), dtype=np.float32)
        n_chunks = int(np.ceil(len(self.test_images) * 1. / self.chunk_size))
        for chunk in xrange(n_chunks):
            # prepare a slice of images to read during this pass
            chunk_end = (chunk + 1) * self.chunk_size
            # we need this to get images if the test set is not divisible by chunk_size
            if len(self.test_images) < chunk_end:
                images = np.zeros((len(self.test_images) - chunk * self.chunk_size,
                                   self.image_size, self.image_size, 3))
                chunk_end = len(self.test_images)
            chunk_slice = slice(chunk * self.chunk_size, chunk_end)
            # read a chunk of images
            for i, image in enumerate(self.test_images[chunk_slice]):
                images[i, ...] = imread(image)
            # change axis order (see comments in valid_gen function) and yield images with labels
            yield np.rollaxis(images, 3, 1)

