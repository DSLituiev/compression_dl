#!/usr/bin/env python3
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import sys
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras
import pickle
import os

batch_size = 128
nb_classes = 10
nb_epoch = 500

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

PERMUTE_Y = True 
PERMUTE_X = -1
URANDOM_X = -1

if PERMUTE_Y:
    infix = "perm_labels_"
    y_train = np.random.permutation(y_train)

if PERMUTE_X>-1:
    infix = "perm_pixels_%d_" % PERMUTE_X
    print(X_train.shape)
    x_file = "dataset/mnist_X_train_permuted_seed%d.pickle" % PERMUTE_X
    X_train = pickle.load(open(x_file, "rb"))
    print(X_train.shape)
    print("using X_train from:\t%s" % x_file)
elif URANDOM_X > -1:
    infix =  "urandom_pixels_%d_" % URANDOM_X
    print(X_train.shape)
    x_file = "dataset/mnist_X_train_urand_states128_seed%d.pickle" % URANDOM_X
    X_train = pickle.load(open(x_file, "rb"))
    print(X_train.shape)
    print("using X_train from:\t%s" % x_file)
else:
    infix = ""


if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)

N_TRAIN = 2000
X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def get_layer_capacities(model):
    layer_shapes = []
    for la in model.layers:
        layer_shapes.append(la.get_output_shape_for(la.input_shape))
            
    capacities = pd.DataFrame({"capacity":[np.prod(ss[1:]) for ss in layer_shapes],
        "layer": np.arange(len(layer_shapes))}).set_index('layer')
    return capacities

def build_model():
    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model


model = build_model()
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


suffix = ( infix + 
          "%d_train_samples" % N_TRAIN
         )
CHECKPOINTS_BASE = "checkpoints/checkpoints_mnist_cnn_" + suffix
CHECKPOINTS_PATH = CHECKPOINTS_BASE + ".weights.{epoch:d}"

print("CHECKPOINTS_PATH:\t", CHECKPOINTS_PATH, file=sys.stderr)

callback_weights = keras.callbacks.ModelCheckpoint(CHECKPOINTS_PATH,
     monitor='val_loss', verbose=0, save_best_only=False,
     save_weights_only=False, mode='auto')


class GetActivations(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        #print(sorted(self.model.__dict__.keys() ))
        #print( sorted(logs.keys()) )
        X = logs["data"][0] #self.model.train_data[0]
        get_layer_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                  [layer_.output for layer_ in  model.layers])

        mode = 1 #  1 - train
        activations = get_layer_outputs([X, mode])
        act_pickle_file = CHECKPOINTS_BASE + '.activations.%d.%d.pickle' % (logs["epoch"], batch)
        with open(act_pickle_file, 'wb') as handle:
            pickle.dump(activations, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del activations
        # get weights
       # weights =  [layer_.params for layer_ in  model.layers]
       # wgt_pickle_file = CHECKPOINTS_BASE + '.weights.%u.pickle' % epoch
       # with open(wgt_pickle_file, 'wb') as handle:
       #     pickle.dump(wgt_pickle_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
       #
        return


def _layer_nnz_(batch_data):
    for layer_n in range(len(batch_data)):
        yield [ np.sum(batch_data[layer_n][sample_n].ravel() != 0.0) for sample_n in range(batch_data[0].shape[0])]
        
def layer_nnz(batch_data):
    return np.vstack(list(_layer_nnz_(batch_data)))


class GetActivationNNZ(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        #print(sorted(self.model.__dict__.keys() ))
        #print( sorted(logs.keys()) )
        X = logs["data"][0] #self.model.train_data[0]
        get_layer_outputs = K.function([model.layers[0].input, K.learning_phase()],
                                  [layer_.output for layer_ in  model.layers])

        mode = 1 #  1 - train
        activations = get_layer_outputs([X, mode])
        act_pickle_file = CHECKPOINTS_BASE + '.nnz.%d.%d.pickle' % (logs["epoch"], batch)
        nnz = layer_nnz(activations)
        with open(act_pickle_file, 'wb') as handle:
            pickle.dump(nnz, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 


get_layer_capacities(model).to_csv(CHECKPOINTS_BASE + ".capacities.tab", sep="\t")

csv_path = CHECKPOINTS_BASE + ".log.csv" 
csv_callback = keras.callbacks.CSVLogger(csv_path, separator=',', append=False)

activation_callback = GetActivationNNZ()


model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test),
          callbacks = [callback_weights, activation_callback, csv_callback])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

sys.path.append(os.path.expanduser("~/infrastructure/"))
from  sendemail import send_ses

fromaddr= "d.lituiev@gmail.com"
recipient=fromaddr
body = "this is a notification from my own AWS instance"
subject = "EC2 Run completed:    %s" % sys.argv[0]

send_ses(fromaddr,
	 subject,
	 body,
	 recipient,
	 attachment=None,
	 filename='')
