__author__ = 'dudevil'

import os.path
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from skimage.io import imread
from utils import lcn_image, global_contrast_normalize, to_ordinal
from multiprocessing import Pool, cpu_count
import theano


def read_files(image_file, image_size=224):
    image = imread(image_file, as_grey=True)
    return image

class DataLoader(object):

    def __init__(self,
                 datadir='data',
                 image_size=224,
                 chunk_size=512,
                 random_state=0,
                 normalize=True,
                 n_jobs=3):
        self.datadir = datadir
        self.image_size = image_size
        self.chunk_size = chunk_size
        self.norm = normalize

        labels = pd.read_csv(os.path.join(self.datadir, "trainLabels.csv"))
        #labels.level = labels.level.clip(upper=1)

        # split the dataset to train and 10% validation (3456 is closest to 10% divisible by batch size 128)
        sss = StratifiedShuffleSplit(labels.level, 1, test_size=1024*3, random_state=random_state)
        #sss = StratifiedKFold(labels.level, 10)
        self.train_index, self.valid_index = list(sss).pop()
        # self.train_index = self.train_index[:1000]
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

        if self.norm:
            self.mean, self.std = self.get_mean_std(self.train_images)

        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        if self.n_jobs:
            self.pool = Pool(self.n_jobs)
        self.random = np.random.RandomState(random_state)


    def valid_gen(self):
        if self.n_jobs:
            return self._batch_iter_parallel(self.valid_images, self.valid_labels)
        else:
            return self._image_iterator(self.valid_images, labels=self.valid_labels)

    def train_gen(self):
        if self.n_jobs:
            return self._batch_iter_parallel(self.train_images, self.train_labels)
        else:
            return self._image_iterator(self.train_images, labels=self.train_labels, transform=True)

    def test_gen(self):
        return self._image_iterator(self.test_images)
        # will leave this as a backup :)
        # images = np.zeros((self.chunk_size, self.image_size, self.image_size, 3), dtype=np.float32)
        # n_chunks = int(np.ceil(len(self.test_images) * 1. / self.chunk_size))
        # for chunk in xrange(n_chunks):
        #     # prepare a slice of images to read during this pass
        #     chunk_end = (chunk + 1) * self.chunk_size
        #     # we need this to get images if the test set is not divisible by chunk_size
        #     if len(self.test_images) < chunk_end:
        #         images = np.zeros((len(self.test_images) - chunk * self.chunk_size,
        #                            self.image_size, self.image_size, 3))
        #         chunk_end = len(self.test_images)
        #     chunk_slice = slice(chunk * self.chunk_size, chunk_end)
        #     # read a chunk of images
        #     for i, image in enumerate(self.test_images[chunk_slice]):
        #         images[i, ...] = imread(image) / 255.
        #     if self.norm:
        #         self.normalize(images)
        #     # change axis order (see comments in valid_gen function) and yield images with labels
        #     yield np.rollaxis(images, 3, 1)

    def _transform(self, image):
        # img = Image.open(image)
        # enhancer = ImageEnhance.Contrast(img)
        # factor = self.random.uniform(.5, 2.)
        # out = np.array(enhancer.enhance(factor), dtype=np.float32) / 255.
        img = imread(image) / 255.
        # flip verticaly with 1/2 probability
        if self.random.randint(2):
            img = img[::-1, ...]
        # flip horizontaly
        if self.random.randint(2):
            img = img[:, ::-1, ...]
        return img


    def _image_iterator(self, image_list, labels=None, transform=False):
        # allocate an array for images
        images = np.zeros((self.chunk_size, self.image_size, self.image_size, 3), dtype=theano.config.floatX)
        n_images = len(image_list)
        n_chunks = n_images // self.chunk_size
        for chunk in xrange(n_chunks):
            # prepare a slice of images to read during this pass
            chunk_end = (chunk + 1) * self.chunk_size
            chunk_slice = slice(chunk * self.chunk_size, chunk_end)
            # read a chunk of images
            for i, image in enumerate(image_list[chunk_slice]):
                if transform:
                    images[i, ...] = self._transform(image)
                else:
                    images[i, ...] = imread(image) / 255.
            if self.norm:
                images = self.normalize(images)
            # change axis order (see comments in valid_gen function) and yield images with labels
            if labels is not None:
                # transform labels to a collumn, but first we need to add a new axis
                #print labels[chunk_slice].values.astype(np.float32).reshape(chunk_slice, 1)
                yield np.rollaxis(images, 3, 1), to_ordinal(labels[chunk_slice].values) #.astype(theano.config.floatX).reshape(len(images), 1)
            else:
                yield np.rollaxis(images, 3, 1)
        # we need to this if the train set size is not divisible by chunk_size
        if n_images > chunk_end:
            imgs_left = n_images - chunk_end
            images = np.zeros((imgs_left, self.image_size, self.image_size, 3), dtype=theano.config.floatX)
            for i, image in enumerate(self.train_images[chunk_end: n_images]):
                if transform:
                    images[i, ...] = self._transform(image)
                else:
                    images[i, ...] = imread(image) / 255.
            if self.norm:
                images = self.normalize(images)
            # change axis order (see comments in valid_gen function) and yield images with labels
            if labels is not None:
                yield np.rollaxis(images, 3, 1), to_ordinal(labels[chunk_end: n_images].values) #.astype(theano.config.floatX).reshape(len(images), 1)
            else:
                yield np.rollaxis(images, 3, 1)

    def get_mean_std(self, images):
        mean = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        mean_sqr = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
        n_images = len(images)
        for image in images:
            img = imread(image) / 255.
            mean += img
            mean_sqr += np.square(img)
        mean = mean / n_images
        std = np.sqrt(mean_sqr / n_images - np.square(mean))
        return mean, std

    def normalize(self, images):
        return (images - self.mean) / (self.std + 1e-5)
