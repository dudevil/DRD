__author__ = 'dudevil'

from lasagne import layers
import lasagne
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from load_dataset import DataLoader

net2 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('conv4', layers.Conv2DLayer),
        ('pool4', layers.MaxPool2DLayer),
        ('conv5', layers.Conv2DLayer),
        ('pool5', layers.MaxPool2DLayer),
        ('conv6', layers.Conv2DLayer),
        ('pool6', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, 224, 224),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_ds=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(3, 3), pool3_ds=(2, 2),
    conv4_num_filters=128, conv4_filter_size=(3, 3), pool4_ds=(2, 2),
    conv5_num_filters=256, conv5_filter_size=(3, 3), pool5_ds=(2, 2),
    conv6_num_filters=256, conv6_filter_size=(3, 3), pool6_ds=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=5, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=1000,
    verbose=1,
    )

dl = DataLoader(image_size=224)
X, y = dl._load_images()  # load 2-d data
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()
#y = ohe.fit_transform(y).toarray()
print(y.shape)
print[y[:20]]
print(X.shape)
net2.fit(lasagne.utils.floatX(X), y)