__author__ = 'dudevil'

import sys
import os.path
import argparse
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from lasagne.utils import floatX
from utils import load_network
from load_dataset import DataLoader


BATCH_SIZE = 128
IMAGE_SIZE = 128

def save_submission(predictions, filenames, n=1):
    assert(len(predictions) == len(filenames))
    names = [os.path.splitext(os.path.basename(image))[0] for image in filenames]
    dfr = pd.DataFrame(data={'image': names, 'level': predictions})
    dfr.to_csv(os.path.join("data", "submissions", "submission_%d.csv" % n), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--network",
                        type=str,
                        default=os.path.join("data", "tidy", "net.pickle"),
                        help="Path to the pickled network file")

    args = parser.parse_args()
    netfile = args.network

    print("Loading saved network...")
    if not os.path.isfile(netfile):
        print("No such file: %s" % netfile)
        exit()
    try:
        network, output = load_network(netfile)
    except Exception, e:
        print("Could not load network: %s" % e)

    print("Loading test dataset...")
    # load test data chunk
    dl = DataLoader(image_size=IMAGE_SIZE)
    test_filenames = dl.test_images
    n_predictions = len(test_filenames)
    print("Compiling theano functions...")
    # set up symbolic variables
    X = T.tensor4('X')
    X_batch = T.tensor4('X_batch')
    batch_index = T.iscalar('batch_index')

    pred = T.iround(output.get_output(X_batch, deterministic=True))
    predict = theano.function(
        [theano.Param(X_batch)],
        pred,
        givens={
            X: X_batch
            },
        )

    print("Predicting...")
    predictions = []
    i = 0
    for test_chunk in dl.test_gen():
        n_batches = int(np.ceil(len(test_chunk) * 1. / BATCH_SIZE))
        for b in xrange(n_batches):
            predictions.append(predict(test_chunk[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]))
        i += 1
        print("%d %%" % (len(predictions) * BATCH_SIZE * 100. / n_predictions))

    print("Saving predictions")
    predictions = np.vstack(predictions)
    save_submission(predictions.flatten(), test_filenames)
