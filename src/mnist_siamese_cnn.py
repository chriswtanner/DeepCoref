'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/keras/')
sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/tensorflow/')

useCNN=False
if sys.argv[1] == "cnn":
    useCNN = True

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    b = [len(digit_indices[d]) for d in range(10)]
    print("b", b)
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    print("n: ",str(n))
    for d in range(10):
        for i in range(n):
            
            # adds positive
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            
            # adds negative
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            
            # adds labels
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    if useCNN:
        seq.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_dim))
        seq.add(Conv2D(64, (3, 3), activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))

        # added following
        seq.add(Conv2D(128, (3, 3), activation='relu'))
        seq.add(Conv2D(256, (3, 3), activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))
        # end of added

        seq.add(Flatten())
        #seq.add(Dense(128, activation='relu'))
        seq.add(Dense(256, activation='relu'))
    else:
        seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(128, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    preds = predictions.ravel() < 0.5
    return ((preds & labels).sum() +
            (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)


print(useCNN)

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train shape1:', x_train.shape)

if useCNN:
    # input image dimensions
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        print("channels first")
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        print("not channels first")
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
# FFNN stuff
else:
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_dim = 784
epochs = 3

if useCNN:
    input_dim = input_shape

print('x_train shape2:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
#print('digit_indices shape:', len(digit_indices))
#print('digit_indices', digit_indices)

training_pairs, training_labels = create_pairs(x_train, digit_indices)

#print(training_pairs)
print('training_pairs:', training_pairs.shape)
print('training_labels:', training_labels.shape)

digit_indices = [np.where(y_test == i)[0] for i in range(10)]
testing_pairs, testing_labels = create_pairs(x_test, digit_indices)

print('testing_pairs:', testing_pairs.shape)
print('testing_labels:', testing_labels.shape)

#print(len(training_pairs[:, 0][0]))
#print(len(training_pairs[:, 1]))
print('training_labels:', training_labels.shape)

# network definition
base_network = create_base_network(input_dim)

if useCNN:
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
else:
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))
print('input_a',input_a.shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

print('distance:', distance.shape)

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
          batch_size=128,
          epochs=epochs,
          validation_data=([testing_pairs[:, 0], testing_pairs[:, 1]], testing_labels))

# compute final accuracy on training and test sets
pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
tr_acc = compute_accuracy(pred, training_labels)
pred = model.predict([testing_pairs[:, 0], testing_pairs[:, 1]])
te_acc = compute_accuracy(pred, testing_labels)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
