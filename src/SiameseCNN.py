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
from ECBHelper import *
from ECBParser import *
import sys
import os

#sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/keras/')
#sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/tensorflow/')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(tf.__version__)

class SiameseCNN:
    def __init__(self, args, corpus, helper):
        print("args:", str(args))
        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        CUDA_VISIBLE_DEVICES=""
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        sess = tf.Session(config=config)
        print(sess)
        print("devices:",device_lib.list_local_devices())

        self.args = args

        self.corpus = corpus
        self.helper = helper

        self.run()

    # trains and tests the model
    def run(self):

        # constructs the training and testing files
        (training_pairs, training_labels), (testing_pairs, testing_labels) = self.helper.createCCNNData()

        input_shape = training_pairs.shape[2:]
        print("input_shape:",str(input_shape))

        print('training_pairs shape1:', training_pairs.shape)

        # input image dimensions
        print('training_pairs shape:', training_pairs.shape)
        print('training_labels shape:', training_labels.shape)
        print('testing pairs shape:', testing_pairs.shape)
        print('testing_labels shape:', testing_labels.shape)

        # network definition
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)
        print('input_a',input_a.shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance,
                          output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss, optimizer=rms, metrics=[self.acc])
        print(model.summary())
        model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
                  batch_size=32,
                  epochs=self.args.numEpochs,
                  validation_data=([testing_pairs[:, 0], testing_pairs[:, 1]], testing_labels))

        # compute final accuracy on training and test sets
        pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
        tr_acc = self.compute_accuracy(pred, training_labels)
        print("training")
        self.compute_f1(pred, training_labels)
        pred = model.predict([testing_pairs[:, 0], testing_pairs[:, 1]])
        te_acc = self.compute_accuracy(pred, testing_labels)
        print("testing")
        self.compute_f1(pred, testing_labels)
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

    # Base network to be shared (eq. to feature extraction).
    def create_base_network(self, input_shape):
        seq = Sequential()
        seq.add(Conv2D(32, kernel_size=(5, 5),activation='relu', input_shape=input_shape, data_format="channels_last"))
        seq.add(Conv2D(64, (5, 5), activation='relu',data_format="channels_first"))
        seq.add(MaxPooling2D(pool_size=(5, 5),data_format="channels_first"))
        seq.add(Dropout(0.25))

        # added following
        '''
        seq.add(Conv2D(128, (3, 3), activation='relu',data_format="channels_first"))
        seq.add(Conv2D(256, (3, 3), activation='relu',data_format="channels_first"))
        seq.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        seq.add(Dropout(0.25))
        # end of added
        '''
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        #seq.add(Dense(256, activation='relu'))
        return seq

    def compute_f1(self, predictions, golds):
        preds = []
        for p in predictions:
            if p < 0.5:
                preds.append(1)
            else:
                preds.append(0)
        
        num_predicted_true = 0
        num_predicted_false = 0
        num_golds_true = 0
        num_tp = 0
        num_correct = 0
        for i in range(len(golds)):
            if golds[i] == 1:
                num_golds_true = num_golds_true + 1

        for i in range(len(preds)):
            if preds[i] == 1:
                num_predicted_true = num_predicted_true + 1
                if golds[i] == 1:
                    num_tp = num_tp + 1
                    num_correct += 1
            else:
                num_predicted_false += 1
                if golds[i] == 0:
                    num_correct += 1
        recall = float(num_tp) / float(num_golds_true)
        prec = 0
        if num_predicted_true > 0:
            prec = float(num_tp) / float(num_predicted_true)
        
        f1 = 0
        if prec > 0 or recall > 0:
            f1 = 2*float(prec * recall) / float(prec + recall)

        accuracy = float(num_correct) / float(len(golds))
        print("------")
        print("num_golds_true: " + str(num_golds_true) + "; num_predicted_false: " + str(num_predicted_false) + "; num_predicted_true: " + str(num_predicted_true) + " (of these, " + str(num_tp) + " actually were)")
        print("recall: " + str(recall) + "; prec: " + str(prec) + "; f1: " + str(f1) + "; accuracy: " + str(accuracy))

    def acc(self, y_true, y_pred):
        ones = K.ones_like(y_pred)
        return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)

    # Compute classification accuracy with a fixed threshold on distances.
    def compute_accuracy(self, predictions, labels):
        preds = predictions.ravel() < 0.5
        return ((preds & labels).sum() +
                (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)