__author__ = 'dudevil'

import numpy as np
import theano
import theano.tensor as T
import time
import lasagne
from lasagne import layers
from load_dataset import DataLoader
from utils import *

IMAGE_SIZE = 224
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MAX_EPOCH = 200


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

conv5 = layers.Conv2DLayer(pool4,
                           num_filters=256,
                           filter_size=(3, 3),
                           W=lasagne.init.Normal())
pool5 = layers.MaxPool2DLayer(conv5, ds=(2, 2))
#
# conv6 = layers.Conv2DLayer(pool5,
#                            num_filters=256,
#                            filter_size=(3, 3),
#                            W=lasagne.init.Orthogonal(gain='relu'))
# pool6 = layers.MaxPool2DLayer(conv6, ds=(2, 2))

dense1 = layers.DenseLayer(pool5,
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

# collect layers to save them later
all_layers = [input, conv1, pool1, conv2, pool2, conv3, pool3, conv4, pool4, conv5, pool5,
              dense1, dense1_dropout, dense2, dense2_dropout, output]


# allocate symbolic variables for theano graph computations
batch_index = T.iscalar('batch_index')
X_batch = T.tensor4('x')
y_batch = T.imatrix('y')

batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

# use mse objective for regression
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
# generate parameter updates for SGD with Nesterov momentum
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

###################
# Actual training #
###################

# keep track of networks best performance and save net configuration
best_epoch = 0
best_kappa = -1.
# epoch and iteration counters
epoch = 0
_iter = 0
# wait for at least this many epochs before saving the model
min_epochs = 50
# store these values for learning curves plotting
train_loss = []
valid_loss = []
kappa_loss = []

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
    # it's quick and dirty
    if not (_iter % 247):
        batch_valid_losses = []
        valid_predictions = []
        # get prediction and error on validation set
        for b in range(num_valid_batches):
            batch_valid_loss, prediction = iter_valid(b)
            batch_valid_losses.append(batch_valid_loss)
            valid_predictions.extend(prediction)
        avg_valid_loss = np.mean(batch_valid_losses)
        c_kappa = kappa(valid_y.get_value(borrow=True), np.array(valid_predictions))
        print("|%6d | %9.6f | %14.6f | %14.5f | %1.3f | %6d |" %
              (epoch,
               avg_train_loss,
               avg_valid_loss,
               c_kappa,
               avg_valid_loss / avg_train_loss,
               time.time() - now))
        # keep track of these for future analysis
        train_loss.append(avg_train_loss)
        valid_loss.append(avg_valid_loss)
        kappa_loss.append(c_kappa)
        # if this is the best kappa obtained so far
        # save the model to make predictions on the test set
        if c_kappa > best_kappa:
            # always wait for min_epochs, to avoid frequent saving
            # during early stages of learning
            if epoch >= min_epochs:
                save_network(all_layers)
            best_kappa = c_kappa
            best_epoch = epoch
        epoch += 1
    if epoch >= MAX_EPOCH:
        break
    _iter += 1
    # load next chunk of data into the GPU
    train_x.set_value(lasagne.utils.floatX(x_next), borrow=True)
    train_y.set_value(y_next, borrow=True)

print("The best weighted quadratic kappa: %.5f obtained on epoch %d.\n The training took %d seconds." %
      (best_kappa, best_epoch, time.time() - now))

results = np.array([train_loss, valid_loss, kappa_loss], dtype=np.float)
np.save("data/tidy/5conv_2dense.npy", results)