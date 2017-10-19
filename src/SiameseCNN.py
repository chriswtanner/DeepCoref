from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import keras
import sys
import os
import math
import operator
import copy
from collections import OrderedDict
from operator import itemgetter
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
from tensorflow.python.client import device_lib
from ECBHelper import *
from ECBParser import *
from get_coref_metrics import *

#sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/keras/')
#sys.path.append('/gpfs/main/home/christanner/.local/lib/python3.5/site-packages/tensorflow/')
#

class SiameseCNN:
    def __init__(self, args, corpus, helper):
        
        self.args = args
        print("args:", str(args))
        print(tf.__version__)

        if args.device == "cpu":
            print("WE WANT TO USE CPU!!")
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            print(sess)
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            #config = tf.ConfigProto(device_count = {'GPU': 0})
            #sess = tf.Session(config=config)
            print(sess)
     
        print("devices:",device_lib.list_local_devices())

        self.corpus = corpus
        self.helper = helper

    # creates clusters for our predictions
    def clusterPredictions(self, pairs, predictions, stoppingPoint):
        clusters = {}
        print("in clusterPredictions()")
        # stores predictions
        docToDMPredictions = defaultdict(lambda : defaultdict(float))
        docToDMs = defaultdict(list) # used for ensuring our predictions included ALL valid DMs
        for i in range(len(pairs)):
            (dm1,dm2) = pairs[i]
            prediction = predictions[i][0]

            doc_id = dm1[0]

            if dm1 not in docToDMs[doc_id]:
                docToDMs[doc_id].append(dm1)
            if dm2 not in docToDMs[doc_id]:
                docToDMs[doc_id].append(dm2)
            docToDMPredictions[doc_id][(dm1,dm2)] = prediction

        ourClusterID = 0
        ourClusterSuperSet = {}

        goldenClusterID = 0
        goldenSuperSet = {}
        
        stoppingPoints = []

        for doc_id in docToDMPredictions.keys():
            print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
            
            # ensures we have all DMs
            if len(docToDMs[doc_id]) != len(self.corpus.docToDMs[doc_id]):
                print("mismatch in DMs!!")
                exit(1)

            #print("docToDMPredictions:",str(docToDMPredictions[doc_id]))
            # sanity check: ensure lower prediction score is good

            sorted_x = sorted(docToDMPredictions[doc_id].items(), key=operator.itemgetter(0))
            bestDM = ""
            bestPred = 9999
            worstDM = ""
            worstPred = -99
            for dmPair in docToDMPredictions[doc_id]:
                pred = docToDMPredictions[doc_id][dmPair]
                if pred < bestPred:
                    bestPred = pred
                    bestDM = dmPair
                if pred > worstPred:
                    worstPred = pred
                    worstDM = dmPair

            
            print("best:",str(bestDM)," had score:",str(bestPred))
            print("\tdm1:",str(self.corpus.dmToMention[bestDM[0]]))
            print("\tdm2:",str(self.corpus.dmToMention[bestDM[1]]))
            print("worst:",str(worstDM)," had score:",str(worstPred))
            print("\tdm1:",str(self.corpus.dmToMention[worstDM[0]]))
            print("\tdm2:",str(self.corpus.dmToMention[worstDM[1]]))

            # construct the golden truth for the current doc
            goldenTruthDirClusters = {}
            for i in range(len(self.corpus.docToREFs[doc_id])):
                tmp = set()
                curREF = self.corpus.docToREFs[doc_id][i]
                for dm in self.corpus.docREFsToDMs[(doc_id,curREF)]:
                    # TMP: 
                    if dm not in self.helper.validDMs:
                        print("skipping:",str(dm))
                        continue
                    tmp.add(dm)
                goldenTruthDirClusters[i] = tmp
                goldenSuperSet[goldenClusterID] = tmp
                goldenClusterID += 1
            #print("golden clusters:", str(goldenTruthDirClusters))
            
            goldenK = len(self.corpus.docToREFs[doc_id])
            print("# golden clusters: ",str(goldenK))
            # constructs our base clusters (singletons)
            ourDirClusters = {}
            for i in range(len(docToDMs[doc_id])):
                dm = docToDMs[doc_id][i]
                if dm not in self.helper.validDMs:
                    print("skipping:",str(dm))
                    continue
                a = set()
                a.add(dm)
                ourDirClusters[i] = a

            #print("golden:",str(goldenTruthDirClusters))


            bestScore = get_conll_f1(goldenTruthDirClusters, ourDirClusters)
            bestClustering = copy.deepcopy(ourDirClusters)

            mergeDistances = []
            f1Scores = []
            mergeDistances.append(-1)
            f1Scores.append(bestScore)

            #print("ourclusters:",str(ourDirClusters))
            print("# initial clusters:",str(len(ourDirClusters.keys()))," had score:",str(bestScore))
            # performs agglomerative, checking our performance after each merge

            while len(ourDirClusters.keys()) > 1:
                # find best merge
                closestDist = 999999
                closestClusterKeys = (-1,-1)

                # looks at all combinations of pairs
                
                for c1 in ourDirClusters.keys():
                    for c2 in ourDirClusters.keys():
                        if c1 == c2:
                            continue

                        for dm1 in ourDirClusters[c1]:
                            for dm2 in ourDirClusters[c2]:
                                dist = 99999
                                if (dm1,dm2) in docToDMPredictions[doc_id]:
                                    dist = docToDMPredictions[doc_id][(dm1,dm2)]
                                elif (dm2,dm1) in docToDMPredictions[doc_id]:
                                    dist = docToDMPredictions[doc_id][(dm2,dm1)]
                                else:
                                    print("* error, why don't we have either dm1 or dm2 in doc_id")
                                if dist < closestDist:
                                    closestDist = dist
                                    closestClusterKeys = (c1,c2)
                                    #print("closestdist is now:",str(closestDist),"which is b/w:",str(closestClusterKeys))
                #print("trying to merge:",str(closestClusterKeys))

                # only merge clusters if it's less than our threshold
                if closestDist > stoppingPoint:
                    break

                mergeDistances.append(closestDist)

                newCluster = set()
                (c1,c2) = closestClusterKeys
                for _ in ourDirClusters[c1]:
                    newCluster.add(_)
                for _ in ourDirClusters[c2]:
                    newCluster.add(_)
                ourDirClusters.pop(c1, None)
                ourDirClusters.pop(c2, None)
                ourDirClusters[c1] = newCluster

                #print("* our updated clusters",str(ourDirClusters))
                curScore = get_conll_f1(goldenTruthDirClusters, ourDirClusters)
                f1Scores.append(curScore)

                #print("\tyielded a score:",str(curScore))
                if curScore > bestScore:
                    #print("(which is a new best!!")
                    bestScore = curScore
                    bestClustering = copy.deepcopy(ourDirClusters)
            
            # end of current doc
            print("best clustering yielded:",str(bestScore),":",str(bestClustering))
            print("# best clusters:",str(len(bestClustering.keys())))
            for i in bestClustering.keys():
                ourClusterSuperSet[ourClusterID] = bestClustering[i]
                ourClusterID += 1

            for i in range(len(f1Scores)):
                if f1Scores[i] == bestScore:
                    print("* ", str(mergeDistances[i])," -> ",str(f1Scores[i]))
                    if i != len(f1Scores) - 1:
                        stoppingPoints.append(mergeDistances[i+1])
                else:
                    print(str(mergeDistances[i])," -> ",str(f1Scores[i]))

        # end of going through every doc
        print("# golden clusters:",str(len(goldenSuperSet.keys())))
        print("# our clusters:",str(len(ourClusterSuperSet)))
        print("stoppingPoints: ",str(stoppingPoints))
        print("avg stopping point: ",str(float(sum(stoppingPoints))/float(len(stoppingPoints))))

        #self.writeCoNLLPerlFile("ourKeys.response",ourClusterSuperSet)
        #self.writeCoNLLPerlFile("ourGolden.keys",goldenSuperSet)      
        #print("finished writing")
        return (ourClusterSuperSet, goldenSuperSet)

    def writeCoNLLPerlFile(self, fileOut, clusters):
        # writes WD file
        f = open(fileOut, 'w')
        f.write("#begin document (t);\n")
        for clusterID in clusters.keys():
            for dm in clusters[clusterID]:
                (doc_id,m_id) = dm
                dirNum = doc_id[0:doc_id.find("_")]
                f.write(str(dirNum) + "\t" + str(doc_id) + ";" + str(m_id) + \
                    "\t(" + str(clusterID) + ")\n")
        f.write("#end document (t);\n")

    # trains and tests the model
    def run(self):

        # loads embeddings
        self.loadEmbeddings(self.args.embeddingsFile, self.args.embeddingsType)

        # constructs the training and dev files
        training_pairs, training_data, training_labels = self.createData(self.helper.trainingDirs, True)
        print("* training data shape:",str(training_data.shape))
        print("* inputA's shape:",str(training_data[:, 0].shape))
        dev_pairs, dev_data, dev_labels = self.createData(self.helper.devDirs, False)

        input_shape = training_data.shape[2:]
        '''
        print("input_shape:",str(input_shape))
        print('training_pairs shape1:', training_pairs.shape)
        print('training_labels shape:', training_labels.shape)
        print('testing pairs shape:', testing_pairs.shape)
        print('testing_labels shape:', testing_labels.shape)
        '''
        # network definition
        print("* input_shape:",str(input_shape))
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
        model.fit([training_data[:, 0], training_data[:, 1]], training_labels,
                  batch_size=self.args.batchSize,
                  epochs=self.args.numEpochs,
                  validation_data=([dev_data[:, 0], dev_data[:, 1]], dev_labels))

        # train accuracy
        print("-----------\npredicting training")
        pred = model.predict([training_data[:, 0], training_data[:, 1]])
        sys.stdout.flush()
        bestProb = self.compute_optimal_f1("training",0.5, pred, training_labels)
        print("training acc:", str(self.compute_accuracy(bestProb, pred, training_labels)))

        '''
        for i in range(len(pairs)):
            gold = "false"
            dm1,dm2 = pairs[i]
            if self.dmToREF[dm1] == self.dmToREF[dm2]:
                gold = "COREF"
            print(str(dm1),str(dm2)," pred:",str(pred[i]), "; gold:", str(gold))
        exit(1)
        '''
        # dev accuracy
        print("-----------\npredicting dev")
        pred = model.predict([dev_data[:, 0], dev_data[:, 1]])
        bestProb = self.compute_optimal_f1("dev", bestProb, pred, dev_labels)
        print("dev acc:", str(self.compute_accuracy(bestProb, pred, dev_labels)))
        # return (dev_pairs, pred)
        
        # clears up ram
        training_pairs = None
        training_data = None
        training_labels = None
        dev_pairs = None
        dev_data = None
        dev_labels = None

        testing_pairs, testing_data, testing_labels = self.createData(self.helper.testingDirs, False)
        print("-----------\npredicting testing")
        pred = model.predict([testing_data[:, 0], testing_data[:, 1]])
        bestProb = self.compute_optimal_f1("testing", bestProb, pred, testing_labels)
        print("test acc:", str(self.compute_accuracy(bestProb, pred, testing_labels)))
        print("testing size:", str(len(testing_data)))

        return (testing_pairs, pred)
        
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
        seq.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
        seq.add(Dropout(0.2))
        seq.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding="same", data_format="channels_first"))
        seq.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
        
        # added following
        if self.args.numLayers == 2:
            print("doing deep!! 2 sections of convolution")
            seq.add(Conv2D(96, (2, 2), activation='relu', padding="same", data_format="channels_first"))
            seq.add(Dropout(0.2))
            seq.add(Conv2D(128, (2, 2), activation='relu', padding="same", data_format="channels_first"))
            seq.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format="channels_first"))
            seq.add(Dropout(0.2))
            # end of added
        elif self.args.numLayers == 3:
            print("doing deeper!! 3 sections of convolution")
            seq.add(Conv2D(96, (2, 2), activation='relu', padding="same", data_format="channels_last"))
            seq.add(Dropout(0.2))
            seq.add(Conv2D(128, (2, 2), activation='relu', padding="same", data_format="channels_last"))
            seq.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format="channels_last"))
            seq.add(Dropout(0.2))
        
            seq.add(Conv2D(196, (2, 2), activation='relu', padding="same", data_format="channels_last"))
            seq.add(Dropout(0.2))
            seq.add(Conv2D(256, (2, 2), activation='relu', padding="same", data_format="channels_last"))
            seq.add(MaxPooling2D(pool_size=(2, 2), padding="same", data_format="channels_last"))
            seq.add(Dropout(0.2))
        seq.add(Flatten())
        if self.args.numLayers == 1:
            seq.add(Dense(128, activation='relu'))
        elif self.args.numLayers == 2:
            seq.add(Dense(256, activation='relu'))
        elif self.args.numLayers == 3:
            seq.add(Dense(512, activation='relu'))
        else:
            print("** ERROR: wrong # of convo layers")
            exit(1)
        return seq

    # from a list of predictions, find the optimal f1 point
    def compute_optimal_f1(self, label, startingProb, predictions, golds):
        print("* in compute_optimal_f1!!!()")
        sys.stdout.flush()
        #print("# preds:",str(len(predictions)))
        # sorts the predictions from smallest to largest
        # (where smallest means most likely a pair)
        preds = set()
        for i in range(len(predictions)):
            preds.add(predictions[i][0])

        #print("# unique preds:",str(len(preds)),flush=True)
        sys.stdout.flush()

        print("< ",str(0.5)," = coref yields:",str(self.compute_f1(0.5, predictions, golds)))

        given = self.compute_f1(startingProb, predictions, golds)
        print("< ",str(startingProb)," = coref yields:",str(given))
        bestProb = startingProb
        bestF1 = given
        
        lowestProb = 0.2
        highestProb = 1.1
        numTried = 0
        #for p in sorted(preds):
        p = lowestProb
        while p < highestProb:
            f1 = self.compute_f1(p, predictions, golds)
            if f1 > bestF1:
                bestF1 = f1
                bestProb = p
            numTried += 1
            p += 0.025
        print(str(label)," BEST F1: ",str(bestProb)," = ", str(bestF1))
        return bestProb

    def compute_f1(self, prob, predictions, golds):
        preds = []
        for p in predictions:
            if p[0] < prob:
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
        #print("------")
        #print("num_golds_true: " + str(num_golds_true) + "; num_predicted_false: " + str(num_predicted_false) + "; num_predicted_true: " + str(num_predicted_true) + " (of these, " + str(num_tp) + " actually were)")
        #print("recall: " + str(recall) + "; prec: " + str(prec) + "; f1: " + str(f1) + "; accuracy: " + str(accuracy))
        return f1

    def acc(self, y_true, y_pred):
        ones = K.ones_like(y_pred)
        return K.mean(K.equal(y_true, ones - K.clip(K.round(y_pred), 0, 1)), axis=-1)

    # Compute classification accuracy with a fixed threshold on distances.
    def compute_accuracy(self, threshold, predictions, labels):
        preds = predictions.ravel() < threshold
        return ((preds & labels).sum() +
                (np.logical_not(preds) & np.logical_not(labels)).sum()) / float(labels.size)

    def loadEmbeddings(self, embeddingsFile, embeddingsType):
        print("* in loadEmbeddings")
        if embeddingsType == "type":
            self.wordTypeToEmbedding = {}
            f = open(embeddingsFile, 'r', encoding="utf-8")
            for line in f:
                tokens = line.rstrip().split(" ")
                wordType = tokens[0]
                emb = [float(x) for x in tokens[1:]]
                self.wordTypeToEmbedding[wordType] = emb
                self.embeddingLength = len(emb)
            f.close()

    # TEMP
    def getCosineSim(self, a, b):
        numerator = 0
        denomA = 0
        denomB = 0
        for i in range(len(a)):
            numerator = numerator + a[i]*b[i]
            denomA = denomA + (a[i]*a[i])
            denomB = denomB + (b[i]*b[i])   
        return float(numerator) / (float(math.sqrt(denomA)) * float(math.sqrt(denomB)))

    def createData(self, dirs, subSample):

        if subSample: # training (we want just some of the negs)
            (pairs, labels) = self.helper.constructSubsampledWDDMPairs(dirs)
        else: # for dev and test (we want all negative examples)
            (pairs, labels) = self.helper.constructAllWDDMPairs(dirs)

        print("# pairs:",str(len(pairs)))
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

                emb = [0]*numCols
                if ind >= 0:
                    token = self.corpus.corpusTokens[ind]
                    if token.text in self.wordTypeToEmbedding:
                        emb = self.wordTypeToEmbedding[token.text]
                    else:
                        print("* ERROR, we don't have:",str(token.text))

                curMentionMatrix[i] = emb

            for i in range(self.args.windowSize):
                ind = t_endIndex + 1 + i

                emb = [0] * numCols
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

        # TEMP; sanity check; just to test if our vectors are constructed correctly
        '''
        added = set()
        x = 0
        for doc in self.corpus.docToDMs:
            print("doc:",str(doc), " has # DMs:", str(len(self.corpus.docToDMs[doc])), " and # REFs:", str(len(self.corpus.docToREFs[doc])))
            for ref in self.corpus.docToREFs[doc]:
                print("\tREF:",str(ref)," has # DMs:", str(len(self.corpus.docREFsToDMs[(doc,ref)])) + ":" + \
                    str(self.corpus.docREFsToDMs[(doc,ref)]))
                for dm1 in self.corpus.docREFsToDMs[(doc,ref)]:
                    print("\t\tDM:",str(dm1)," text:",str(self.corpus.dmToMention[dm1].text))
                    cosineScores = {}
                    v1 = dmToMatrix[dm1][0]
                    for dm2 in self.corpus.docToDMs[doc]:
                        if dm1 == dm2:
                            continue
                        v2 = dmToMatrix[dm2][0]
                        cs = self.getCosineSim(v1,v2)
                        cosineScores[dm2] = cs
                    sorted_distances = sorted(cosineScores.items(), key=operator.itemgetter(1), reverse=True)
                    for _ in sorted_distances:
                        dm3 = _[0]
                        if self.corpus.dmToREF[dm3] == self.corpus.dmToREF[dm1]:
                            print ("\t\t\t***", str(_), str(self.corpus.dmToMention[dm3].text))
                        else:
                            print("\t\t\t",str(_), str(self.corpus.dmToMention[dm3].text))
        '''
        # constructs final 5D matrix
        X = []
        for (dm1,dm2) in pairs:
            pair = np.asarray([dmToMatrix[dm1],dmToMatrix[dm2]])
            X.append(pair)
        Y = np.asarray(labels)
        X = np.asarray(X)

        return (pairs, X,Y)
