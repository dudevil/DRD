__author__ = 'dudevil'

import os.path
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from skimage.io import imread
from skimage.transform import rotate
from utils import lcn_image, global_contrast_normalize
from multiprocessing import Pool, cpu_count, Process, Queue
from functools import partial
import theano


def read_files(image_file, image_size=224):
    image = imread(image_file) / 255.
    return image

def _transform(rng, image):
    # img = Image.open(image)
    # enhancer = ImageEnhance.Contrast(img)
    # factor = self.random.uniform(.5, 2.)
    # out = np.array(enhancer.enhance(factor), dtype=np.float32) / 255.
    img = imread(image) / 255.
    # flip verticaly with 1/2 probability
    if rng.randint(2):
        img = img[::-1, ...]
    # flip horizontaly
    if rng.randint(2):
        img = img[:, ::-1, ...]
    return img[np.newaxis, ...]


class Worker(Process):

    def __init__(self,
                 images,
                 labels,
                 batch_queue,
                 batch_size=64,
                 random_state=42,
                 mean=None,
                 std=None,
                 augment=True,
                 shuffle=True):
        super(Worker, self).__init__()
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.outqueue = batch_queue
        self.rng = np.random.RandomState(random_state)
        self.mean = mean
        self.std = std
        self.augment = augment
        self.shuffle = shuffle

    def _transform(self, image):
        img = imread(image) / 255.
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= (self.std + 1e-5)
        # flip verticaly with 1/2 probability
        if self.augment:
            img = rotate(img, self.rng.randint(0, 30))
            if self.rng.randint(2):
                img = img[::-1, ...]
            # flip horizontaly
            if self.rng.randint(2):
                img = img[:, ::-1, ...]
        return img[np.newaxis, ...]

    def run(self):
        while True:
            for i in xrange(0, len(self.images), self.batch_size):
                if self.augment:
                    transformed = map(self._transform, self.images[i:i+self.batch_size])
                else:
                    transformed = map(lambda img: (imread(img) / 255.)[np.newaxis, ...],
                                      self.images[i:i+self.batch_size])
                transformed = np.vstack(transformed)
                self.outqueue.put((np.rollaxis(transformed, 3, 1),
                                   self.labels[i:i+self.batch_size].values.astype(np.int32)))
            self.outqueue.put(None)
            if self.shuffle:
                shuffle_idx = self.rng.permutation(len(self.labels))
                self.images = self.images.iloc[shuffle_idx]
                self.labels = self.labels.iloc[shuffle_idx]

class DataLoader(object):

    def __init__(self,
                 datadir='data',
                 image_size=224,
                 random_state=0,
                 batch_size=64,
                 parallel=True,
                 normalize=True):

        self.datadir = datadir
        self.image_size = image_size
        self.norm = normalize
        self.random = np.random.RandomState(random_state)
        self.batch_size = batch_size
        self.parallel = parallel
        labels = pd.read_csv(os.path.join(self.datadir, "trainLabels.csv"))

        # split the dataset to train and 10% validation (3456 is closest to 10% divisible by batch size 128)
        sss = StratifiedShuffleSplit(labels.level, 1, test_size=1024*3, random_state=random_state)
        #sss = StratifiedKFold(labels.level, 10)
        self.train_index, self.valid_index = list(sss).pop()
        #self.train_index = self.train_index[:1000]

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

        if parallel:
            self.train_queue = Queue(10)
            self.valid_queue = Queue(10)
        # get mean and std across training set
        if self.norm:
            self.mean, self.std = self.get_mean_std(self.train_images)
        else:
            self.mean = self.std = None

        self.train_worker = Worker(self.train_images,
                                   self.train_labels,
                                   self.train_queue,
                                   batch_size=self.batch_size,
                                   mean=self.mean,
                                   std=self.std)
        self.valid_worker = Worker(self.valid_images,
                                   self.valid_labels,
                                   self.valid_queue,
                                   batch_size=self.batch_size,
                                   mean=self.mean,
                                   std=self.std,
                                   augment=False,
                                   shuffle=False)
        self.train_worker.start()
        self.valid_worker.start()

    def valid_gen(self):
        if self.parallel:
            return iter(self.valid_queue.get, None)
        return self._image_iterator(self.valid_images, labels=self.valid_labels)

    def train_gen(self):
        if self.parallel:
            return iter(self.train_queue.get, None)
        shuffle_idx = self.random.permutation(len(self.train_labels))
        return self._image_parallel(self.train_images.iloc[shuffle_idx],
                                    labels=self.train_labels.iloc[shuffle_idx],
                                    transform=True)

    def test_gen(self):
        return self._image_iterator(self.test_images)

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
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3), dtype=theano.config.floatX)
        n_images = len(image_list)
        n_chunks = n_images // self.batch_size
        for chunk in xrange(n_chunks):
            # prepare a slice of images to read during this pass
            chunk_end = (chunk + 1) * self.batch_size
            chunk_slice = slice(chunk * self.batch_size, chunk_end)
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
                yield np.rollaxis(images, 3, 1), labels[chunk_slice].values.astype('int32')
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
                yield np.rollaxis(images, 3, 1), labels[chunk_end: n_images].values.astype('int32')
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

    # broken code, leaving as backup
    # def _image_parallel(self, images, labels=None, transform=False):
    #     batch_size = self.batch_size
    #     res = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=theano.config.floatX)
    #     if transform:
    #         f = partial(_transform, self.random)
    #     else:
    #         f = read_files
    #     for i, im in enumerate(self.pool.imap(f, images, chunksize=batch_size)):
    #         res[i % batch_size, ...] = im
    #         if (i + 1) % batch_size == 0:
    #             if labels is not None:
    #                 yield np.rollaxis(res, 3, 1), labels[i - batch_size + 1: i + 1].values.astype('int32')
    #             else:
    #                 yield np.rollaxis(res, 3, 1)
    #     imgs_left = (i + 1) % batch_size
    #     if imgs_left:
    #         if labels is not None:
    #             yield np.rollaxis(res[:imgs_left], 3, 1), labels[-imgs_left:].values.astype('int32')
    #         else:
    #             yield np.rollaxis(res[:imgs_left], 3, 1)
