__author__ = 'dudevil'

import os.path
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from skimage.io import imread
from skimage import transform
from utils import lcn_image, global_contrast_normalize, to_ordinal
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
                 pseudo_images='',
                 pseudo_labels='',
                 proportion=0.,
                 augment=False):
        super(Worker, self).__init__()
        assert len(images) == len(labels)
        self.daemon = True
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.outqueue = batch_queue
        self.rng = np.random.RandomState(random_state)
        self.mean = mean
        self.std = std
        self.augment = augment
        if proportion:
            assert len(pseudo_images) == len(pseudo_labels)
            self.pseudo_images = images
            self.pseudo_labels = pseudo_labels
            if isinstance(proportion, float):
                self.n_pseudo = int(self.batch_size * proportion)
            elif isinstance(proportion, int):
                self.n_pseudo = proportion
            else:
                raise ValueError("Pseudo-labels proportion must be either a float or an int.")
        else:
            self.n_pseudo = 0

    def _transform(self, image):
        img = imread(image) / 255.
          # add color augmentation from: http://arxiv.org/ftp/arxiv/papers/1501/1501.02876.pdf
        # normalize images if mean and std were specified
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= (self.std + 1e-5)
        if self.augment:
            #img = rotate(img, self.rng.randint(-30, 30))
            # flip verticaly with 1/2 probability
            if self.rng.randint(2):
                img = img[::-1, ...]
            # flip horizontaly
            if self.rng.randint(2):
                img = img[:, ::-1, ...]
            r, g, b = self.rng.randint(2, size=3)
            if r:
                img[:, :, 0] = img[:, :, 0] + self.rng.randint(-30, 30)/255.
            if g:
                img[:, :, 1] = img[:, :, 1] + self.rng.randint(-30, 30)/255.
            if b:
                img[:, :, 2] = img[:, :, 2] + self.rng.randint(-30, 30)/255.

            # random shifts
            # shift_x = self.rng.randint(-4, 4)
            # shift_y = self.rng.randint(-4, 4)
            # shift = transform.SimilarityTransform(translation=[shift_x, shift_y])
            # img = transform.warp(img, shift, mode='constant', cval=0.0)

        return img[np.newaxis, ...]

    def run(self):
        n_true = self.batch_size - self.n_pseudo
        if self.n_pseudo:
            test_idx = np.arange(len(self.pseudo_images))
        while True:
            # send images through queue in batches
            for i in xrange(0, len(self.images), n_true):
                # read real train images
                batch = map(self._transform, self.images[i: i + n_true])
                if self.n_pseudo:
                    pseudo_idx = self.rng.choice(test_idx, self.n_pseudo)
                    batch.extend(map(self._transform, self.pseudo_images.iloc[pseudo_idx]))
                    labels = np.vstack((
                        to_ordinal(self.labels[i: i + n_true].values),
                        self.pseudo_labels[pseudo_idx].astype(theano.config.floatX)
                    ))
                else:
                    labels = to_ordinal(self.labels[i: i + n_true].values)
                    #labels = self.labels[i: i + n_true].values.astype(np.int32)
                batch = np.vstack(batch)
                self.outqueue.put((np.rollaxis(batch, 3, 1), labels))
                #self.outqueue.put((batch.reshape(len(batch), 1 , 128, 128), labels))
            # shuffle images at epoch end do this for trainig set only
            if self.augment:
                shuffle_idx = self.rng.permutation(len(self.images))
                self.images = self.images.iloc[shuffle_idx]
                self.labels = self.labels.iloc[shuffle_idx]
            # signal end of epoch
            self.outqueue.put(None)


class DataLoader(object):

    def __init__(self,
                 image_size=224,
                 random_state=0,
                 batch_size=64,
                 pseudo_proportion=0.,
                 parallel=True,
                 normalize=True,
                 datadir="data",
                 train_path=os.path.join("train", "resized"),
                 test_path=os.path.join("test", "resized"),):
        train_path = os.path.join(datadir, train_path)
        test_path = os.path.join(datadir, test_path)
        self.image_size = image_size
        self.norm = normalize
        self.random = np.random.RandomState(random_state)
        self.batch_size = batch_size
        self.parallel = parallel
        labels = pd.read_csv(os.path.join(datadir, "trainLabels.csv"))
        # get only levels 0,1,2
        # labels = labels[labels.level < 3]
        # labels.level[labels.level == 2] = 1
        # split the dataset to train and 10% validation (3456 is closest to 10% divisible by batch size 128)
        sss = StratifiedShuffleSplit(labels.level, 1, test_size=0.1, random_state=random_state)
        self.train_index, self.valid_index = list(sss).pop()
        #self.train_index = self.train_index[:1000]

        # get train and validation labels
        self.train_labels = labels.level[self.train_index]
        self.valid_labels = labels.level[self.valid_index]
        # prepare train and test image files
        self.train_images = labels.image[self.train_index].apply(lambda img:
                                                                 os.path.join(train_path, img + ".png"))
        self.valid_images = labels.image[self.valid_index].apply(lambda img:
                                                                 os.path.join(train_path, img + ".png"))
        self.test_images = [os.path.join(test_path, img) for img in os.listdir(test_path)]
        pseudos = pd.read_csv("data/submissions/submission_90.csv")
        self.pseudo_images = pseudos.iloc[:, 0].apply(lambda img: os.path.join(test_path, img + ".png"))
        self.pseudo_labels = pseudos.iloc[:, 1:].values

        if self.norm:
            self.mean, self.std = self.get_mean_std(self.train_images)
            # this code leads to a weird GPU-driver error needs further investigation
            # mean_file = os.path.join(train_path, "mean.npy")
            # std_file = os.path.join(train_path, "std.npy")
            # # if we saved mean and std earlier grab them from files
            # if os.path.isfile(mean_file) and os.path.isfile(std_file):
            #     print("loading saved mean")
            #     self.mean = np.load(mean_file)
            #     self.std = np.load(std_file)
            # else:
            #     # calculate mean and std by iterating over the trainset
            #     self.mean, self.std = self.get_mean_std(self.train_images)
            #     # save mean and std for future use in this directory
            #     np.save(mean_file, self.mean)
            #     np.save(std_file, self.std)
            #     print("calculated and saved")
        else:
            self.mean = self.std = None

        if parallel:
            self.train_queue = Queue(10)
            self.valid_queue = Queue(10)
            # get mean and std across training set
            if pseudo_proportion:
                self.train_worker = Worker(self.train_images,
                                           self.train_labels,
                                           self.train_queue,
                                           pseudo_images=self.pseudo_images,
                                           pseudo_labels=self.pseudo_labels,
                                           proportion=0.05,
                                           batch_size=self.batch_size,
                                           mean=self.mean,
                                           std=self.std,
                                           augment=True)
            else:
                self.train_worker = Worker(self.train_images,
                                           self.train_labels,
                                           self.train_queue,
                                           batch_size=self.batch_size,
                                           mean=self.mean,
                                           std=self.std,
                                           augment=True)
            self.valid_worker = Worker(self.valid_images,
                                       self.valid_labels,
                                       self.valid_queue,
                                       batch_size=self.batch_size,
                                       mean=self.mean,
                                       std=self.std,
                                       augment=False)
            self.train_worker.start()
            self.valid_worker.start()

    def valid_gen(self):
        if self.parallel:
            return iter(self.valid_queue.get, None)
        return self._image_iterator(self.valid_images, labels=self.valid_labels)

    def train_gen(self):
        if self.parallel:
            return iter(self.train_queue.get, None)
        else:
            # shuffle images on each epoch
            shuffle_idx = self.random.permutation(len(self.train_labels))
            return self._image_iterator(self.train_images.iloc[shuffle_idx],
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
        n_chunks = int(np.ceil(n_images * 1. / self.batch_size))
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
                yield np.rollaxis(images, 3, 1), to_ordinal(labels[chunk_slice].values)
            else:
                yield np.rollaxis(images, 3, 1)
        # we need to this if the train set size is not divisible by batch_size
        # if n_images > chunk_end:
        #     imgs_left = n_images - chunk_end
        #     images = np.zeros((imgs_left, self.image_size, self.image_size, 3), dtype=theano.config.floatX)
        #     for i, image in enumerate(self.train_images[chunk_end: n_images]):
        #         if transform:
        #             images[i, ...] = self._transform(image)
        #         else:
        #             images[i, ...] = imread(image) / 255.
        #     if self.norm:
        #         images = self.normalize(images)
        #     # change axis order (see comments in valid_gen function) and yield images with labels
        #     if labels is not None:
        #         yield np.rollaxis(images, 3, 1), to_ordinal(labels[chunk_end: n_images].values) #.astype(theano.config.floatX).reshape(len(images), 1)
        #     else:
        #         yield np.rollaxis(images, 3, 1)

    def cleanup(self):
        # terminate the worker processes
        if self.parallel:
            self.train_worker.terminate()
            self.valid_worker.terminate()

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
