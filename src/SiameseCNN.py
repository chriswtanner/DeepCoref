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
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(tf.__version__)

class SiameseCNN:
    def __init__(self, args, corpus, helper):
        self.args = args
        print("args:", str(args))
        
        # GPU stuff
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print(sess)
        #config = tf.ConfigProto(
        #    device_count = {'GPU': 0}
        #)
        #sess = tf.Session(config=config)
        #print(sess)
        print("devices:",device_lib.list_local_devices())

        self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22]
        self.devDirs = [23,24,25]
        self.testingDirs = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

        self.corpus = corpus
        self.helper = helper

        self.run()

    # trains and tests the model
    def run(self):

        # loads embeddings
        self.loadEmbeddings(self.args.embeddingsFile, self.args.embeddingsType)

        # constructs the training and dev files
        (training_pairs, training_labels) = self.createData(self.trainingDirs)
        (dev_pairs, dev_labels) = self.creatData(self.devDirs)

        input_shape = training_pairs.shape[2:]
        '''
        print("input_shape:",str(input_shape))
        print('training_pairs shape1:', training_pairs.shape)
        print('training_labels shape:', training_labels.shape)
        print('testing pairs shape:', testing_pairs.shape)
        print('testing_labels shape:', testing_labels.shape)
        '''
        # network definition
        base_network = self.create_base_network(input_shape)

        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        model = Model([input_a, input_b], distance)

        # train
        rms = RMSprop()
        model.compile(loss=self.contrastive_loss, optimizer=rms)
        print(model.summary())
        model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_labels,
                  batch_size=self.args.batchSize,
                  epochs=self.args.numEpochs,
                  validation_data=([dev_pairs[:, 0], dev_pairs[:, 1]], dev_labels))

        # compute final accuracy on training and test sets
        print("predicting training")
        pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
        self.compute_optimal_f1(pred, training_labels)

        # clears up ram
        training_pairs = None
        training_labels = None
        dev_pairs = None
        dev_labels = None

        (testing_pairs, testing_labels) = self.creatData(self.testingDirs)
        print("predicting testing")
        pred = model.predict([testing_pairs[:, 0], testing_pairs[:, 1]])
        self.compute_optimal_f1(pred, testing_labels)
        #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        #print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

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
        seq.add(Conv2D(128, (4, 4), activation='relu'))
        seq.add(Conv2D(256, (3, 3), activation='relu',data_format="channels_first"))
        seq.add(MaxPooling2D(pool_size=(2, 2),data_format="channels_first"))
        seq.add(Dropout(0.25))
        # end of added
        
        seq.add(Flatten())
        seq.add(Dense(128, activation='relu'))
        #seq.add(Dense(256, activation='relu'))
        return seq

    # from a list of predictions, find the optimal f1 point
    def compute_optimal_f1(self, predictions, golds):

        # sorts the predictions from smallest to largest
        # (where smallest means most likely a pair)
        valToDMPair = defaultdict(list)
        for i in range(predictions):
            DMPair = "a"
            val = predictions[i]
            valToDMPair[val].append(DMPair)
        print("# of unique vals:",len(valToDMPair.keys()))

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
        print("* computing accuracy")
        preds = predictions.ravel() < 0.5
        return ((preds & labels).sum() +
                (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)

    def constructDMPairs(self, dirs):
        pairs = []
        labels = []
        for dirNum in sorted(self.corpus.dirToREFs.keys()):
            if dirNum not in dirs:
                continue
            dirDMs = []
            for ref in self.corpus.dirToREFs[dirNum]:
                for dm in self.corpus.refToDMs[ref]:
                    dirDMs.append(dm)
            added = set()
            for dm1 in dirDMs:
                for dm2 in dirDMs:
                    if dm1 == dm2 or (dm1,dm2) in added or (dm2,dm1) in added:
                        continue

                    pairs.append((dm1,dm2))
                    if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]:
                        labels.append(1)
                    else:
                        labels.append(0) 

                    added.add((dm1,dm2))
                    added.add((dm2,dm1))
        return (pairs, labels)


TODO: when and how am i calling this fuction
    def constructTrainingPairs(self):
        print("* in constructListsOfTrainingPairs")
        trainingPositives = []
        trainingNegatives = []

        for dirNum in sorted(self.corpus.dirToREFs.keys()):

            # only process the training dirs
            if dirNum > self.trainingCutoff:
                continue

            added = set() # so we don't add the same pair twice

            numRefsForThisDir = len(self.corpus.dirToREFs[dirNum]) 
            for i in range(numRefsForThisDir):
                ref1 = self.corpus.dirToREFs[dirNum][i]
                for dm1 in self.corpus.refToDMs[ref1]:
                    for dm2 in self.corpus.refToDMs[ref1]:
                        if (dm1,dm2) not in added and (dm2,dm1) not in added:
                            # adds a positive example
                            trainingPositives.append((dm1,dm2))
                            added.add((dm1,dm2))
                            added.add((dm2,dm1))

                            numNegsAdded = 0
                            j = i + 1
                            while numNegsAdded < self.args.numNegPerPos:
                                ref2 = self.corpus.dirToREFs[dirNum][j%numRefsForThisDir]
                                if ref2 == ref1:
                                    continue
                                numDMs = len(self.corpus.refToDMs[ref2])
                                dm3 = self.corpus.refToDMs[ref2][randint(0, numDMs-1)]
                                #if (dm1,dm3) not in added and (dm3,dm1) not in added:
                                trainingNegatives.append((dm1,dm3))
                                    #added.add((dm1,dm3))
                                    #added.add((dm3,dm1))
                                numNegsAdded += 1
                                j += 1
                                
        # shuffle training
        if self.args.shuffleTraining:
            numPositives = len(trainingPositives)
            for i in range(numPositives):
                # pick 2 to change in place
                a = randint(0,numPositives-1)
                b = randint(0,numPositives-1)
                swap = trainingPositives[a]
                trainingPositives[a] = trainingPositives[b]
                trainingPositives[b] = swap

            numNegatives = len(trainingNegatives)
            for i in range(numNegatives):
                # pick 2 to change in place
                a = randint(0,numNegatives-1)
                b = randint(0,numNegatives-1)
                swap = trainingNegatives[a]
                trainingNegatives[a] = trainingNegatives[b]
                trainingNegatives[b] = swap

        print("#pos:",str(len(trainingPositives)))
        print("#neg:",str(len(trainingNegatives)))
        trainingPairs = []
        trainingLabels = []
        j = 0
        for i in range(len(trainingPositives)):
            trainingPairs.append(trainingPositives[i])
            trainingLabels.append(1)
            for _ in range(self.args.numNegPerPos):
                trainingPairs.append(trainingNegatives[j])
                trainingLabels.append(0)
                j+=1
        return (trainingPairs,trainingLabels)

    def loadEmbeddings(self, embeddingsFile, embeddingsType):
        print("* in loadEmbeddings")
        if embeddingsType == "type":
            self.wordTypeToEmbedding = {}
            f = open(embeddingsFile, 'r')
            for line in f:
                tokens = line.rstrip().split(" ")
                wordType = tokens[0]
                emb = [float(x) for x in tokens[1:]]
                self.wordTypeToEmbedding[wordType] = emb
                self.embeddingLength = len(emb)
            f.close()

    def createData(self, dirs):

        (pairs, labels) = self.constructDMPairs(dirs)

        # constructs the DM matrix for every mention
        dmToMatrix = {}

        numRows = 1 + 2*self.args.windowSize
        numCols = self.embeddingLength
        for m in self.corpus.mentions:
            
            curMentionMatrix = np.zeros(shape=(numRows,numCols))
            #print("mention:",str(m))
            t_startIndex = 99999999
            t_endIndex = -1

            # gets token indices and constructs the Mention embedding
            menEmbedding = [0]*numCols
            for t in m.corpusTokenIndices:
                token = self.corpus.corpusTokens[t]

                curEmbedding = self.wordTypeToEmbedding[token.text]
                menEmbedding = [x + y for x,y in zip(menEmbedding, curEmbedding)]

                ind = self.corpus.corpusTokensToCorpusIndex[token]
                if ind < t_startIndex:
                    t_startIndex = ind
                if ind > t_endIndex:
                    t_endIndex = ind

            # sets the center
            curMentionMatrix[self.args.windowSize] = [x / float(len(m.corpusTokenIndices)) for x in menEmbedding]

            # the prev tokens
            for i in range(self.args.windowSize):
                ind = t_startIndex - self.args.windowSize + i

                emb = [0]*50
                if ind >= 0:
                    token = self.corpus.corpusTokens[ind]
                    if token.text in self.wordTypeToEmbedding:
                        emb = self.wordTypeToEmbedding[token.text]
                    else:
                        print("* ERROR, we don't have:",str(token.text))

                curMentionMatrix[i] = emb

            for i in range(self.args.windowSize):
                ind = t_endIndex + 1 + i

                emb = [0] * 50
                if ind < self.corpus.numCorpusTokens - 1:
                    token = self.corpus.corpusTokens[ind]
                    #print("next",str(token))
                    if token.text in self.wordTypeToEmbedding:
                        emb = self.wordTypeToEmbedding[token.text]
                    else:
                        print("* ERROR, we don't have:",str(token.text))
                curMentionMatrix[self.args.windowSize+1+i] = emb
            curMentionMatrix = np.asarray(curMentionMatrix).reshape(numRows,numCols,1)

            dmToMatrix[(m.doc_id,int(m.m_id))] = curMentionMatrix

        # constructs final 5D matrix
        X = []
        for (dm1,dm2) in pairs:
            pair = np.asarray([dmToMatrix[dm1],dmToMatrix[dm2]])
            X.append(pair)
        Y = np.asarray(labels)
        X = np.asarray(X)

        return (X,Y)