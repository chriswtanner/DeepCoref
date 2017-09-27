from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from tensorflow.python.client import device_lib
import sys
sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/keras/')
sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/tensorflow/')

class SiameseCNN:
    def __init__(self, args): # , corpus, helper):
        print("we in here")
        print("args:", str(args))
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess)
        print("devices:",device_lib.list_local_devices())
        #self.corpus = corpus
        #self.helper = helper
        self.run()

    # trains and tests the model
    def run(self):

        # loads training and test
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('x_train shape1:', x_train.shape)

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

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        epochs = 3

        print('x_train shape2:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_test shape:', y_test.shape)
        # create training+test positive and negative pairs
        digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        #print('digit_indices shape:', len(digit_indices))
        #print('digit_indices', digit_indices)

        training_pairs, training_labels = self.create_pairs(x_train, digit_indices)

        #print(training_pairs)
        print('training_pairs:', training_pairs.shape)
        print('training_labels:', training_labels.shape)

        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        testing_pairs, testing_labels = self.create_pairs(x_test, digit_indices)

        print('testing_pairs:', testing_pairs.shape)
        print('testing_labels:', testing_labels.shape)

        #print(len(training_pairs[:, 0][0]))
        #print(len(training_pairs[:, 1]))
        print('training_labels:', training_labels.shape)

        # network definition
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        print('input_a',input_a.shape)

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        print('distance:', distance.shape)

        model = Model([input_a, input_b], distance)

        # train
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss, optimizer=rms)
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

    def euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)


    # Contrastive loss from Hadsell-et-al.'06
    # http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    def contrastive_loss(self, y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    # Positive and negative pair creation.
    # Alternates between positive and negative pairs.
    def create_pairs(self, x, digit_indices):
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

    # Base network to be shared (eq. to feature extraction).
    def create_base_network(self, input_shape):
        seq = Sequential()
        seq.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=input_shape))
        seq.add(Conv2D(64, (3, 3), activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))

        # added following
        '''
        seq.add(Conv2D(128, (3, 3), activation='relu'))
        seq.add(Conv2D(256, (3, 3), activation='relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.25))
        # end of added
        '''

        seq.add(Flatten())
        #seq.add(Dense(128, activation='relu'))
        seq.add(Dense(256, activation='relu'))
        return seq

    # Compute classification accuracy with a fixed threshold on distances.
    def compute_accuracy(self, predictions, labels):
        preds = predictions.ravel() < 0.5
        return ((preds & labels).sum() +
                (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)


'''


'''