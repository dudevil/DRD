import lasagne
from lasagne import layers, nonlinearities
from lasagne.layers import dnn
from custom_layers import SliceRotateLayer, RotateMergeLayer, leaky_relu


IMAGE_SIZE = 128
BATCH_SIZE = 64
MOMENTUM = 0.9

input = layers.InputLayer(shape=(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))

slicerot = SliceRotateLayer(input)

conv1 = dnn.Conv2DDNNLayer(slicerot,
                           num_filters=64,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool1 = dnn.MaxPool2DDNNLayer(conv1, (3, 3), stride=(2, 2))

conv2_dropout = lasagne.layers.DropoutLayer(pool1, p=0.1)
conv2 = dnn.Conv2DDNNLayer(conv2_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool2 = dnn.MaxPool2DDNNLayer(conv2, (3, 3), stride=(2, 2))

conv3_dropout = lasagne.layers.DropoutLayer(pool2, p=0.1)
conv3 = dnn.Conv2DDNNLayer(conv3_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)

conv4_dropout = lasagne.layers.DropoutLayer(conv3, p=0.1)
conv4 = dnn.Conv2DDNNLayer(conv4_dropout,
                           num_filters=128,
                           filter_size=(3, 3),
                           W=lasagne.init.Orthogonal(gain='relu'),
                           nonlinearity=leaky_relu)
pool4 = dnn.MaxPool2DDNNLayer(conv4, (3, 3), stride=(2, 2))

conv5_dropout = lasagne.layers.DropoutLayer(pool4, p=0.1)
conv5 = dnn.Conv2DDNNLayer(conv5_dropout,
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

dense1_dropout = lasagne.layers.DropoutLayer(merge, p=0.5)
dense1a = layers.DenseLayer(dense1_dropout,
                            num_units=2048,
                            W=lasagne.init.Normal(),
                            nonlinearity=None)

dense1 = layers.FeaturePoolLayer(dense1a, 2)

dense2_dropout = lasagne.layers.DropoutLayer(dense1, p=0.5)
dense2a = layers.DenseLayer(dense2_dropout,
                            num_units=2048,
                            W=lasagne.init.Normal(),
                            nonlinearity=None)
dense2 = layers.FeaturePoolLayer(dense2a, 2)


out_dropout = lasagne.layers.DropoutLayer(dense2, p=0.5)
output = layers.DenseLayer(out_dropout,
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
              dense1_dropout, dense1a, dense1,
              dense2_dropout, dense2a, dense2,
              out_dropout, output]