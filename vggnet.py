__author__ = 'dudevil'

import os
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
import lasagne
from lasagne import layers
from skimage.io import imread_collection, imread
from sklearn.metrics import confusion_matrix
from load_dataset import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MAX_EPOCH = 100

def normalize(imgs, std_reg=1e-5):
    """
    Performs per-pixel zero-mean unit varience standartization of images
    :param imgs:
    :param std_reg:
    :return:
    """
    return (imgs - imgs.mean(axis=0, keepdims=True)) / (imgs.std(axis=0, keepdims=True) + std_reg)

#def negative_log_likelihood(y_true, y_pred):
#    return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

def kappa(y_true, y_pred):
    """
    Quadratic kappa score: http://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation
    Implementaion mostly taken from: http://scikit-learn-laboratory.readthedocs.org/en/latest/_modules/skll/metrics.html

    :param y_true:
    :param y_pred:
    :return:
    """
    # Ensure that the lists are both the same length
    assert(len(y_true) == len(y_pred))

    # This rather crazy looking typecast is intended to work as follows:
    # If an input is an int, the operations will have no effect.
    # If it is a float, it will be rounded and then converted to an int
    # because the ml_metrics package requires ints.
    # If it is a str like "1", then it will be converted to a (rounded) int.
    # If it is a str that can't be typecast, then the user is
    # given a hopefully useful error message.
    # Note: numpy and python 3.3 use bankers' rounding.
    try:
        y_true = [int(np.round(float(y))) for y in y_true]
        y_pred = [int(np.round(float(y))) for y in y_pred]
    except ValueError as e:
        print("Kappa values must be integers or strings")
        raise e

    # Figure out normalized expected values
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = [y - min_rating for y in y_true]
    y_pred = [y - min_rating for y in y_pred]

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = confusion_matrix(y_true, y_pred,
                                labels=list(range(num_ratings)))
    num_scored_items = float(len(y_true))

    # Build weight array if weren't passed one
    weights = np.empty((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = abs(i - j) ** 2

    hist_true = np.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[: num_ratings] / num_scored_items
    hist_pred = np.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[: num_ratings] / num_scored_items
    expected = np.outer(hist_true, hist_pred)

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    k = 1.0
    if np.count_nonzero(weights):
        k -= (sum(sum(weights * observed)) / sum(sum(weights * expected)))

    return k

print("Loading dataset...")
dloader = DataLoader(image_size=IMAGE_SIZE)

# get train data chunk and load it into GPU
train_x, train_y = dloader.train_gen().next()
num_train_batches = len(train_x) // BATCH_SIZE
train_x = theano.shared(lasagne.utils.floatX(train_x))
train_y = theano.shared(train_y)
# get validation data chunk and load it into GPU
valid_x, valid_y = dloader.train_gen().next()
num_valid_batches = len(valid_x) // BATCH_SIZE
valid_x = theano.shared(lasagne.utils.floatX(valid_x))
valid_y = theano.shared(valid_y)

#####################
#  Build the model  #
#####################
print("Building model...")
input = layers.InputLayer(shape=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

conv1 = layers.Conv2DLayer(input,
                           num_filters=32,
                           filter_size=(3, 3),
                           W=lasagne.init.Normal())
pool1 = layers.MaxPool2DLayer(conv1, ds=(2, 2))

conv2 = layers.Conv2DLayer(pool1,
                           num_filters=64,
                           filter_size=(3, 3),
                           W=lasagne.init.Normal())
pool2 = layers.MaxPool2DLayer(conv2, ds=(2, 2))

conv3 = layers.Conv2DLayer(pool2,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Normal())
pool3 = layers.MaxPool2DLayer(conv3, ds=(2, 2))

conv4 = layers.Conv2DLayer(pool3,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Normal())
pool4 = layers.MaxPool2DLayer(conv4, ds=(2, 2))

# conv5 = layers.Conv2DLayer(pool4,
#                            num_filters=256,
#                            filter_size=(3, 3),
#                            W=lasagne.init.Orthogonal(gain='relu'))
# pool5 = layers.MaxPool2DLayer(conv5, ds=(2, 2))
#
# conv6 = layers.Conv2DLayer(pool5,
#                            num_filters=256,
#                            filter_size=(3, 3),
#                            W=lasagne.init.Orthogonal(gain='relu'))
# pool6 = layers.MaxPool2DLayer(conv6, ds=(2, 2))

dense1 = layers.DenseLayer(pool4,
                           num_units=2048,
                           W=lasagne.init.Normal())
dense1_dropout = lasagne.layers.DropoutLayer(dense1, p=0.5)

dense2 = layers.DenseLayer(dense1_dropout,
                           num_units=2048,
                           W=lasagne.init.Normal())
dense2_dropout = lasagne.layers.DropoutLayer(dense2, p=0.5)

output = layers.DenseLayer(dense2_dropout,
                           num_units=1,
                           nonlinearity=None)


# allocate symbolic variables for theano graph computations
batch_index = T.iscalar('batch_index')
X_batch = T.tensor4('x')
y_batch = T.imatrix('y')

batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

objective = lasagne.objectives.Objective(output,
                                         loss_function=lasagne.objectives.mse)

loss_train = objective.get_loss(X_batch, target=y_batch)
loss_eval = objective.get_loss(X_batch, target=y_batch,
                               deterministic=True)
# calculates actual predictions to determine weighted kappa
# http://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation

pred = T.iround(output.get_output(X_batch, deterministic=True))

# collect all model parameters
all_params = lasagne.layers.get_all_params(output)
# generated parameter updates for SGD with Nesterov momentum
updates = lasagne.updates.nesterov_momentum(
    loss_train, all_params, LEARNING_RATE, MOMENTUM)

print("Compiling theano functions...")
# create theano functions for calculating losses on train and validation sets
iter_train = theano.function(
    [batch_index], loss_train,
    updates=updates,
    givens={
        X_batch: train_x[batch_slice],
        y_batch: train_y[batch_slice],
        },
    )
iter_valid = theano.function(
    [batch_index], [loss_eval, pred],
    givens={
        X_batch: valid_x[batch_slice],
        y_batch: valid_y[batch_slice],
        },
    )

best_iter = 0
best_valid_loss = np.inf
epoch = 0
_iter = 0

now = time.time()
print("| Epoch | Train err | Validation err | Weighted Kappa | Ratio |  Time  |")
print("|----------------------------------------------------------------------|")

# get next chunks of data
for x_next, y_next in dloader.train_gen():
    batch_train_losses = []
    # perform forward pass and parameters update
    for b in range(num_train_batches):
        batch_train_loss = iter_train(b)
        batch_train_losses.append(batch_train_loss)

    avg_train_loss = np.mean(batch_train_losses)

    # with batch_size = 128 an epoch takes about 247 iterations
    # we measure validation performance once per epoch
    if not (_iter % 247):
        batch_valid_losses = []
        valid_predictions = []
        # get prediction and error on validation set
        for b in range(num_valid_batches):
            batch_valid_loss, prediction = iter_valid(b)
            batch_valid_losses.append(batch_valid_loss)
            valid_predictions.extend(prediction)
        avg_valid_loss = np.mean(batch_valid_losses)
        print("|%6d | %9.6f | %14.6f | %14.6f | %1.3f | %6d |" %
              (epoch,
               avg_train_loss,
               avg_valid_loss,
               kappa(train_y.get_value(borrow=True), np.array(valid_predictions)),
               avg_valid_loss / avg_train_loss,
               time.time() - now))
        epoch += 1
    if epoch >= MAX_EPOCH:
        break
    _iter += 1
    # load next chunk of data into the GPU
    train_x.set_value(lasagne.utils.floatX(x_next), borrow=True)
    train_y.set_value(y_next, borrow=True)