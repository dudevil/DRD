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
from custom_layers import SliceRotateLayer, RotateMergeLayer, leaky_relu
from load_dataset import DataLoader
from sklearn.metrics import confusion_matrix
from utils import *


IMAGE_SIZE = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.02
MOMENTUM = 0.9
MAX_EPOCH = 130
#LEARNING_RATE_SCHEDULE = np.logspace(-5.6, -10, MAX_EPOCH, base=2., dtype=theano.config.floatX)
LEARNING_RATE_SCHEDULE = {110: 0.001,
                          120: 0.0005}

print("Loading dataset...")
dloader = DataLoader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, random_state=16, train_path="train/trimmed")

#####################
#  Build the model  #
#####################
print("Building model...")

input = layers.InputLayer(shape=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

slicerot = SliceRotateLayer(input)

conv1 = layers.Conv2DLayer(slicerot,
                           num_filters=64,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool1 = dnn.MaxPool2DDNNLayer(conv1, (3, 3), stride=(2, 2))

#conv2_dropout = lasagne.layers.DropoutLayer(pool1, p=0.1)
conv2 = layers.Conv2DLayer(pool1,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool2 = dnn.MaxPool2DDNNLayer(conv2, (3, 3), stride=(2, 2))

#conv3_dropout = lasagne.layers.DropoutLayer(pool2, p=0.1)
conv3 = layers.Conv2DLayer(pool2,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)

#conv4_dropout = lasagne.layers.DropoutLayer(conv3, p=0.1)
conv4 = layers.Conv2DLayer(conv3,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool4 = dnn.MaxPool2DDNNLayer(conv4, (3, 3), stride=(2, 2))

#conv5_dropout = lasagne.layers.DropoutLayer(pool4, p=0.1)
conv5 = layers.Conv2DLayer(pool4,
                           num_filters=256,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
# conv6_dropout = lasagne.layers.DropoutLayer(conv5, p=0.1)
# conv6 = layers.Conv2DLayer(conv6_dropout,
#                            num_filters=256,
#                            filter_size=(3, 3),
#                            W=lasagne.init.Orthogonal(gain='relu'))
pool6 = dnn.MaxPool2DDNNLayer(conv5, (2, 2), stride=(2, 2))

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
              conv2, pool2,
              conv3,
              conv4, pool4,
              conv5, pool6,
              merge,
              dense1a, dense1, dense1_dropout,
              dense2a, dense2, dense2_dropout,
              output]
print_network(all_layers)

# allocate symbolic variables for theano graph computations
batch_index = T.iscalar('batch_index')
X_batch = T.tensor4('x')
y_batch = T.fmatrix('y')

# allocate shared variables for images, labels and learing rate
x_shared = theano.shared(np.zeros((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=theano.config.floatX),
                         borrow=True)
y_shared = theano.shared(np.zeros((BATCH_SIZE, 4), dtype=theano.config.floatX),
                         borrow=True)
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
pred = T.gt(output.get_output(X_batch, deterministic=True), 0.5)

#pred = T.cast(output.get_output(X_batch, deterministic=True), 'int32').clip(0, 4)
# collect all model parameters
all_params = lasagne.layers.get_all_params(output)
# generate parameter updates for SGD with Nesterov momentum
updates = lasagne.updates.nesterov_momentum(
    loss_train, all_params, LEARNING_RATE, MOMENTUM)


print("Compiling theano functions...")
# create theano functions for calculating losses on train and validation sets
iter_train = theano.function(
    [], loss_train,
    updates=updates,
    givens={
        X_batch: x_shared,
        y_batch: y_shared,
        },
    )
iter_valid = theano.function(
    [], [loss_eval, pred],
    givens={
        X_batch: x_shared,
        y_batch: y_shared,
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
min_epochs = 30
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
        if epoch in LEARNING_RATE_SCHEDULE:
            learning_rate.set_value(LEARNING_RATE_SCHEDULE[epoch])
        epoch += 1
        # train the network on all chunks
        batch_train_losses = []
        for x_next, y_next in dloader.train_gen():
            # perform forward pass and parameters update
            if not len(x_next) == BATCH_SIZE:
                continue
            x_shared.set_value(lasagne.utils.floatX(x_next), borrow=True)
            y_shared.set_value(y_next, borrow=True)
            batch_train_loss = iter_train()
            batch_train_losses.append(batch_train_loss)
            #num_train_batches = int(np.ceil(len(x_next) / BATCH_SIZE))
        avg_train_loss = np.mean(batch_train_losses)
        # validate the network on validation chunks
        batch_valid_losses = []
        valid_predictions = []
        # get prediction and error on validation set
        for valid_x_next, valid_y_next in dloader.valid_gen():
            #print valid_y_next
            x_shared.set_value(lasagne.utils.floatX(valid_x_next), borrow=True)
            y_shared.set_value(valid_y_next, borrow=True)
            batch_valid_loss, prediction = iter_valid()
            batch_valid_losses.append(batch_valid_loss)
            valid_predictions.extend(get_predictions(prediction, batch_size=BATCH_SIZE))
            #num_valid_batches = len(valid_x_next) // BATCH_SIZE
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

elapsed_time = time.time() - now
print("The best weighted quadratic kappa: %.5f obtained on epoch %d.\n The training took %d seconds." %
      (best_kappa, best_epoch, elapsed_time))
print(" The average performance was %.1f images/sec" % (
    (len(dloader.train_images) + len(dloader.valid_images)) * float(epoch) / elapsed_time))

results = np.array([train_loss, valid_loss, kappa_loss], dtype=np.float)
np.save("data/tidy/training.npy", results)
np.save("data/tidy/confusion.npy", conf_mat)
imgs_error.to_csv("data/tidy/imgs_error.csv")
# terminate background tasks
dloader.cleanup()