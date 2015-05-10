import numpy as np
import theano
import theano.tensor as T
import time
import pandas as pd
import functools
import operator
import lasagne
from lasagne.layers import dnn
from lasagne import layers, regularization, nonlinearities
from custom_layers import SliceRotateLayer, RotateMergeLayer, FractionalPool2DLayer, StochasticPoolLayer
from load_dataset import DataLoader
from sklearn.metrics import confusion_matrix
from utils import *


IMAGE_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.02
MOMENTUM = 0.9
MAX_EPOCH = 100
LEARNING_RATE_SCHEDULE = np.logspace(-5.6, -10, MAX_EPOCH, base=2., dtype=theano.config.floatX)

print("Loading dataset...")
dloader = DataLoader(image_size=IMAGE_SIZE, n_jobs=0, chunk_size=64, normalize=True, random_state=42)
# get train data chunk and load it into GPU
train_x, train_y = dloader.train_gen().next()
num_train_batches = len(train_x) // BATCH_SIZE
train_x = theano.shared(lasagne.utils.floatX(train_x), borrow=True)
train_y = theano.shared(train_y, borrow=True)
# get validation data chunk and load it into GPU
valid_x, valid_y = dloader.valid_gen().next()
num_valid_batches = len(valid_x) // BATCH_SIZE
valid_x = theano.shared(lasagne.utils.floatX(valid_x), borrow=True)
valid_y = theano.shared(valid_y, borrow=True)

#####################
#  Build the model  #
#####################
print("Building model...")


input = layers.InputLayer(shape=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

slicerot = SliceRotateLayer(input)

conv1 = layers.Conv2DLayer(slicerot,
                           num_filters=64,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'))
#pool1 = dnn.MaxPool2DDNNLayer(conv1, (3, 3), stride=(2, 2))
pool1 = FractionalPool2DLayer(conv1, (np.sqrt(2), np.sqrt(2)))

conv2_dropout = lasagne.layers.DropoutLayer(pool1, p=0.1)
conv2 = layers.Conv2DLayer(conv2_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'))
pool2 = FractionalPool2DLayer(conv2, (np.sqrt(2), np.sqrt(2)))
#pool2 = dnn.MaxPool2DDNNLayer(conv2, (3, 3), stride=(2, 2))
#pool2 = StochasticPoolLayer(conv2, (2, 2))

conv3_dropout = lasagne.layers.DropoutLayer(pool2, p=0.1)
conv3 = layers.Conv2DLayer(conv3_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'))

conv4_dropout = lasagne.layers.DropoutLayer(conv3, p=0.1)
conv4 = layers.Conv2DLayer(conv4_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'))
pool4 = dnn.MaxPool2DDNNLayer(conv4, (3, 3), stride=(2, 2))
#pool4 = FractionalPool2DLayer(conv4, (np.sqrt(2), np.sqrt(2)))

conv5_dropout = lasagne.layers.DropoutLayer(pool4, p=0.1)
conv5 = layers.Conv2DLayer(conv5_dropout,
                           num_filters=256,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'))

# conv6_dropout = lasagne.layers.DropoutLayer(conv5, p=0.1)
# conv6 = layers.Conv2DLayer(conv6_dropout,
#                            num_filters=256,
#                            filter_size=(3, 3),
#                            W=lasagne.init.Orthogonal(gain='relu'))
pool6 = dnn.MaxPool2DDNNLayer(conv5, (2, 2), stride=(2, 2))
#pool6 = FractionalPool2DLayer(conv5, (np.sqrt(2), np.sqrt(2)))
merge = RotateMergeLayer(pool6)

dense1a = layers.DenseLayer(merge,
                            num_units=2048,
                            W=lasagne.init.Normal(),
                            nonlinearity=None)

dense1 = layers.FeaturePoolLayer(dense1a, 2)
dense1_dropout = lasagne.layers.DropoutLayer(dense1, p=0.5)

dense2a = layers.DenseLayer(dense1_dropout,
                            num_units=2048,
                            W=lasagne.init.Normal(),
                            nonlinearity=None)

dense2 = layers.FeaturePoolLayer(dense2a, 2)
dense2_dropout = lasagne.layers.DropoutLayer(dense2, p=0.5)

output = layers.DenseLayer(dense2_dropout,
                           num_units=4,
                           nonlinearity=nonlinearities.sigmoid)

# collect layers to save them later
all_layers = [input,
              slicerot,
              conv1, pool1,
              conv2_dropout, conv2, pool2,
              conv3_dropout, conv3,
              conv4_dropout, conv4, pool4,
              conv5_dropout, conv5, pool6,
              merge,
              dense1a, dense1, dense1_dropout,
              dense2a, dense2, dense2_dropout,
              output]

# allocate symbolic variables for theano graph computations
batch_index = T.iscalar('batch_index')
X_batch = T.tensor4('x')
y_batch = T.fmatrix('y')
learning_rate = theano.shared(np.float32(LEARNING_RATE))

batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

# use mse objective for regression
objective = lasagne.objectives.Objective(output,
                                         loss_function=lasagne.objectives.mse)

loss_train = objective.get_loss(X_batch, target=y_batch) #+ 0.05 * (
    # regularization.l2(dense1) + regularization.l2(dense2) + regularization.l2(conv1) +
    # regularization.l2(conv2) + regularization.l2(conv3)
#)
loss_eval = objective.get_loss(X_batch, target=y_batch,
                               deterministic=True)

# calculates actual predictions to determine weighted kappa
# http://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation
#pred = T.argmax(output.get_output(X_batch, deterministic=True), axis=1)
pred = T.round(T.sum(output.get_output(X_batch, deterministic=True), axis=1))

#pred = T.cast(output.get_output(X_batch, deterministic=True), 'int32').clip(0, 4)
# collect all model parameters
all_params = lasagne.layers.get_all_params(output)
# generate parameter updates for SGD with Nesterov momentum
updates = lasagne.updates.nesterov_momentum(
    loss_train, all_params, LEARNING_RATE, MOMENTUM)

for layer in all_layers:
    output_shape = layer.get_output_shape()
    print("  {:<18}\t{:<20}\tproduces {:>7} outputs".format(
        layer.__class__.__name__,
        str(output_shape),
        str(functools.reduce(operator.mul, output_shape[1:])),
        ))

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
best_valid = 1.
best_kappa = 0.
# epoch and iteration counters
epoch = 0
_iter = 0
# wait for at least this many epochs before saving the model
min_epochs = 10
# store these values for learning curves plotting
train_loss = []
valid_loss = []
kappa_loss = []
conf_mat = np.array([])
imgs_error = pd.Series([])
# wait for this many epochs if the validation error is not increasing
patience = 10
now = time.time()
print("| Epoch | Train err | Validation err | Weighted Kappa | Ratio |  Time  |")
print("|----------------------------------------------------------------------|")

try:
    # get next chunks of data
    while epoch < MAX_EPOCH:
        learning_rate.set_value(LEARNING_RATE_SCHEDULE[epoch])
        epoch += 1
        # train the network on all chunks
        batch_train_losses = []
        for x_next, y_next in dloader.train_gen():
            # perform forward pass and parameters update
            for b in xrange(num_train_batches):
                batch_train_loss = iter_train(b)
                batch_train_losses.append(batch_train_loss)
            train_x.set_value(lasagne.utils.floatX(x_next), borrow=True)
            train_y.set_value(y_next, borrow=True)
            num_train_batches = int(np.ceil(len(x_next) / BATCH_SIZE))
        avg_train_loss = np.mean(batch_train_losses)
        # validate the network on validation chunks
        batch_valid_losses = []
        valid_predictions = []
        # get prediction and error on validation set
        for valid_x_next, valid_y_next in dloader.valid_gen():
            #print valid_y_next
            for b in xrange(num_valid_batches):
                batch_valid_loss, prediction = iter_valid(b)
                batch_valid_losses.append(batch_valid_loss)
                valid_predictions.extend(prediction)
            valid_x.set_value(lasagne.utils.floatX(valid_x_next), borrow=True)
            valid_y.set_value(valid_y_next, borrow=True)
            num_valid_batches = len(valid_x_next) // BATCH_SIZE
        avg_valid_loss = np.mean(batch_valid_losses)
        vp = np.array(valid_predictions)
        c_kappa = kappa(dloader.valid_labels, vp)
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
            conf_mat = confusion_matrix(dloader.valid_labels, valid_predictions)
            imgs_error = images_byerror(valid_predictions, dloader.valid_labels.values, dloader.valid_images.values)
            best_kappa = c_kappa
            best_epoch = epoch
            patience = 10
        else:
            #decrease patience
            patience -= 1
except KeyboardInterrupt:
    print("Trainig interrupted on epoch %d" % epoch)

print("The best weighted quadratic kappa: %.5f obtained on epoch %d.\n The training took %d seconds." %
      (best_kappa, best_epoch, time.time() - now))

results = np.array([train_loss, valid_loss, kappa_loss], dtype=np.float)
np.save("data/tidy/training.npy", results)
np.save("data/tidy/confusion.npy", conf_mat)
imgs_error.to_csv("data/tidy/imgs_error.csv")