import numpy as np
import theano
import theano.tensor as T
import time
import argparse
import lasagne
import os
from lasagne import layers, regularization, nonlinearities
from load_dataset import DataLoader
from sklearn.metrics import confusion_matrix
from utils import *

import sys

IMAGE_SIZE = 256
BATCH_SIZE = 32
MOMENTUM = 0.9
MAX_EPOCH = 1
#LEARNING_RATE_SCHEDULE = dict(enumerate(np.logspace(-5.6, -10, MAX_EPOCH, base=2., dtype=theano.config.floatX)))
LEARNING_RATE_SCHEDULE = {
    0: 0.02,
    130: 0.01,
    140: 0.005,
    150: 0.002,
    160: 0.001,
    170: 0.0005,
    180: 0.0002,
    190: 0.0001,
    }

if __name__ == '__main__':
    #####################
    # Get cmd arguments #
    #####################
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--network",
                        type=str,
                        help="Path to the pickled network file")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default='',
                        help="Path to the file storing network configuration")
    parser.add_argument("-e",
                        "--epochs",
                        type=int,
                        help="Number of epochs to train the network")
    args = parser.parse_args()

    print("Loading dataset...")
    dloader = DataLoader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, random_state=1106, train_path="train/trimmed256")

    # for Rasim    
    #dloader = DataLoader(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, random_state=16, datadir="C:/workspace/projects/kaggle/retina-diabetic")

    #####################
    #  Build the model  #
    #####################
    if args.model:
        execfile(args.model)
        print("Built model:")
    elif args.network:
        all_layers, output = load_network(args.network)
        print("Loaded network: ")

    # if command-line argument was specified it overrides default and config MAX_EPOCH
    if args.epochs:
        MAX_EPOCH = args.epochs

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
    learning_rate = theano.shared(np.float32(LEARNING_RATE_SCHEDULE[0]))
    
    # use mse objective for regression
    # objective = lasagne.objectives.MaskedObjective(output,
    #                                                loss_function=lasagne.objectives.mse,
    #                                                aggregation='sum')
    objective = lasagne.objectives.Objective(output,
                                             loss_function=lasagne.objectives.mse)
    mask = np.array([1, 2, 3, 4], dtype=theano.config.floatX)
    loss_train = objective.get_loss(X_batch, target=y_batch)
    
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)
    
    # calculates actual predictions to determine weighted kappa
    # http://www.kaggle.com/c/diabetic-retinopathy-detection/details/evaluation
    #pred = T.argmax(output.get_output(X_batch, deterministic=True), axis=1)
    probas = lasagne.layers.get_output(output, X_batch, deterministic=True)
    pred = T.gt(probas, 0.5)
    
    #pred = T.cast(output.get_output(X_batch, deterministic=True), 'int32').clip(0, 4)
    # collect all model parameters
    all_params = lasagne.layers.get_all_params(output)
    # generate parameter updates for SGD with Nesterov momentum
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, MOMENTUM)

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
            #chunk_num = 0
            for valid_x_next, valid_y_next in dloader.valid_gen():
                # probas = np.zeros((4, valid_x_next.shape[0], 4), dtype=theano.config.floatX)
                if not len(valid_x_next) == BATCH_SIZE:
                    continue
                x_shared.set_value(lasagne.utils.floatX(valid_x_next), borrow=True)
                y_shared.set_value(valid_y_next, borrow=True)
                batch_valid_loss, prediction = iter_valid()
                batch_valid_losses.append(batch_valid_loss)
                valid_predictions.extend(get_predictions(prediction))
            avg_valid_loss = np.mean(batch_valid_losses)
            vp = np.array(valid_predictions)
            #print valid_predictions
            #print dloader.valid_labels
            c_kappa = np.sum(valid_predictions == dloader.valid_labels.values) / float(len(dloader.valid_labels))
            #kappa(dloader.valid_labels, vp)
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
                    imgs_error = make_predictions_series(valid_predictions, dloader.valid_images.values)
                    #imgs_error = images_byerror(valid_predictions, dloader.valid_labels.values, dloader.valid_images.values)
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
