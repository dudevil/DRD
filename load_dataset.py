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
                 random_state=0):
        self.datadir = datadir
        self.random_state = random_state
        self.image_size = image_size
        self.chunk_size = 128

        labels = pd.read_csv(os.path.join(self.datadir, "trainLabels.csv"))
        # split the dataset to train and 10% validation
        sss = StratifiedShuffleSplit(labels.level, 1, test_size=0.1, random_state=random_state)
        self.train_index, self.valid_index = list(sss).pop()
        # get train and validation labels
        self.train_labels = labels.level[self.train_index]
        self.valid_labels = labels.level[self.valid_index]
        # prepare train and test image files
        self.train_images = labels.image[self.train_index].apply(lambda img:
                                                                 os.path.join("data", "train", "resized", img + ".jpg"))
        self.valid_images = labels.image[self.valid_index].apply(lambda img:
                                                                 os.path.join("data", "train", "resized", img + ".jpg"))

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
        n_chunks = len(self.train_images) // self.chunk_size
        while True:
            for chunk in xrange(n_chunks):
                # prepare a slice of images to read during this pass
                chunk_slice = slice(chunk * self.chunk_size, (chunk + 1) * self.chunk_size)
                # read a chunk of images
                for i, image in enumerate(self.train_images[chunk_slice]):
                    images[i, ...] = imread(image)
                # change axis order (see comments in valid_gen function) and yield images with labels
                yield np.rollaxis(images, 3, 1), self.train_labels[chunk_slice].values.astype(np.int32)[np.newaxis].T