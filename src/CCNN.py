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
from keras.layers import Dense, Activation, Dropout, merge, Merge, Flatten, Input, Lambda, Conv2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import RMSprop, Adagrad, Adam
from keras import backend as K
from tensorflow.python.client import device_lib
from ECBHelper import *
from ECBParser import *
from get_coref_metrics import *
from array import array

class CCNN:
    def __init__(self, args, corpus, helper, hddcrp_parsed, isWDModel):
        self.calculateMax = False # find the max pairwise performance
        self.useRelationalFeatures = False # the merged layer right before final output
        self.NNBasic = False # if True, instead of a CCNN, actually just use a FF
        self.args = args

        print("args:", str(args))
        print("tf version:",str(tf.__version__))

        if args.device == "cpu":
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("session:",str(sess))
        print("devices:",device_lib.list_local_devices())

        self.corpus = corpus
        self.helper = helper
        self.hddcrp_parsed = hddcrp_parsed
        self.isWDModel = isWDModel # if True, do our original model; if False, do CD training

        # just for understanding the data more
        self.lemmas = set()
        self.OOVLemmas = set()
        self.mentionLengthToMentions = defaultdict(list)
        
        # used for mapping multi-token mentions to a canonical mention representation
        #self.mentionLemmaTokenCounts = defaultdict(int)
        print("-----------------------")

    # creates clusters for our predictions
    def clusterPredictions(self, pairs, predictions, stoppingPoint):

        if isWDModel:
            clusters = {}
            print("in clusterPredictions() -- WD Model")
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
                #print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
                
                # ensures we have all DMs
                if len(docToDMs[doc_id]) != len(self.corpus.docToDMs[doc_id]):
                    print("mismatch in DMs!!")
                    exit(1)

                # construct the golden truth for the current doc
                goldenTruthDirClusters = {}
                for i in range(len(self.corpus.docToREFs[doc_id])):
                    tmp = set()
                    curREF = self.corpus.docToREFs[doc_id][i]
                    for dm in self.corpus.docREFsToDMs[(doc_id,curREF)]:
                        # TMP:
                        if False: #self.args.runOnValid:
                            if dm not in self.helper.validDMs:
                                print("skipping:",str(dm))
                                continue
                        
                        tmp.add(dm)
                    goldenTruthDirClusters[i] = tmp
                    goldenSuperSet[goldenClusterID] = tmp
                    goldenClusterID += 1
                #print("golden clusters:", str(goldenTruthDirClusters))
                
                goldenK = len(self.corpus.docToREFs[doc_id])
                #print("# golden clusters: ",str(goldenK))
                # constructs our base clusters (singletons)
                ourDocClusters = {}
                for i in range(len(docToDMs[doc_id])):
                    dm = docToDMs[doc_id][i]
                    if False: #self.args.runOnValid:
                        if dm not in self.helper.validDMs:
                            print("skipping:",str(dm))
                            continue
                    
                    a = set()
                    a.add(dm)
                    ourDocClusters[i] = a

                #print("golden:",str(goldenTruthDirClusters))
                # the following keeps merging until our shortest distance > stopping threshold,
                # or we have 1 cluster, whichever happens first
                if not self.calculateMax:
                    while len(ourDocClusters.keys()) > 1:
                        # find best merge
                        closestDist = 999999
                        closestClusterKeys = (-1,-1)

                        closestAvgDist = 999999
                        closestAvgClusterKeys = (-1,-1)

                        closestAvgAvgDist = 999999
                        closestAvgAvgClusterKeys = (-1,-1)

                        #print("ourDocClusters:",str(ourDocClusters.keys()))
                        # looks at all combinations of pairs
                        i = 0
                        for c1 in ourDocClusters.keys():
                            
                            #print("c1:",str(c1))
                            j = 0
                            for c2 in ourDocClusters.keys():
                                if j > i:
                                    avgavgdists = []
                                    for dm1 in ourDocClusters[c1]:
                                        avgdists = []
                                        for dm2 in ourDocClusters[c2]:
                                            dist = 99999
                                            if (dm1,dm2) in docToDMPredictions[doc_id]:
                                                dist = docToDMPredictions[doc_id][(dm1,dm2)]
                                                avgavgdists.append(dist)
                                                avgdists.append(dist)
                                            elif (dm2,dm1) in docToDMPredictions[doc_id]:
                                                dist = docToDMPredictions[doc_id][(dm2,dm1)]
                                                avgavgdists.append(dist)
                                                avgdists.append(dist)
                                            else:
                                                print("* error, why don't we have either dm1 or dm2 in doc_id")
                                                exit(1)
                                            if dist < closestDist:
                                                closestDist = dist
                                                closestClusterKeys = (c1,c2)  
                                        avgDist = float(sum(avgdists)) / float(len(avgdists))
                                        if avgDist < closestAvgDist:
                                            closestAvgDist = avgDist
                                            closestAvgClusterKeys = (c1,c2)
                                    avgavgDist = float(sum(avgavgdists)) / float(len(avgavgdists))
                                    if avgavgDist < closestAvgAvgDist:
                                        closestAvgAvgDist = avgavgDist
                                        closestAvgAvgClusterKeys = (c1,c2)
                                j += 1
                            i += 1

                                                #print("closestdist is now:",str(closestDist),"which is b/w:",str(closestClusterKeys))
                            #print("trying to merge:",str(closestClusterKeys))

                        # only merge clusters if it's less than our threshold
                        #if closestDist > stoppingPoint:
                        # changed
                        if self.args.clusterMethod == "min" and closestDist > stoppingPoint:
                            break
                        elif self.args.clusterMethod == "avg" and closestAvgDist > stoppingPoint:
                            break
                        elif self.args.clusterMethod == "avgavg" and closestAvgAvgDist > stoppingPoint:
                            break

                        newCluster = set()
                        (c1,c2) = closestClusterKeys
                        if self.args.clusterMethod == "avg":
                            (c1,c2) = closestAvgClusterKeys
                        elif self.args.clusterMethod == "avgavg":
                            (c1,c2) = closestAvgAvgClusterKeys

                        for _ in ourDocClusters[c1]:
                            newCluster.add(_)
                        for _ in ourDocClusters[c2]:
                            newCluster.add(_)
                        ourDocClusters.pop(c1, None)
                        ourDocClusters.pop(c2, None)
                        ourDocClusters[c1] = newCluster
                    # end of current doc
                    for i in ourDocClusters.keys():
                        ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
                        #print("setting ourClusterSuperSet[",str(ourClusterID),"] to:",str(ourDocClusters[i]))
                        ourClusterID += 1

            # end of going through every doc
            print("# total golden clusters:",str(len(goldenSuperSet.keys())))
            print("# total our clusters:",str(len(ourClusterSuperSet)))
            #print("stoppingPoints: ",str(stoppingPoints))
            #print("avg stopping point: ",str(float(sum(stoppingPoints))/float(len(stoppingPoints))))

            #self.writeCoNLLPerlFile("ourKeys.response",ourClusterSuperSet)
            #self.writeCoNLLPerlFile("ourGolden.keys",goldenSuperSet)      
            #print("finished writing")
            return (ourClusterSuperSet, goldenSuperSet)
        else: # working w/ CD model
            clusters = {}
            print("in clusterPredictions() -- CD Model")
            # stores predictions
            dirHalfToDMPredictions = defaultdict(lambda : defaultdict(float))
            dirHalfToDMs = defaultdict(list) # used for ensuring our predictions included ALL valid DMs
            for i in range(len(pairs)):
                (dm1,dm2) = pairs[i]
                prediction = predictions[i][0]


                doc_id1 = dm1[0]
                extension1 = doc_id1[doc_id1.find("ecb"):]
                dir_num1 = int(doc_id1.split("_")[0])

                doc_id2 = dm2[0]
                extension2 = doc_id2[doc_id2.find("ecb"):]
                dir_num2 = int(doc_id2.split("_")[0])

                if extension1 != extension2 or dir_num1 != dir_num2:
                    print("** ERROR: we are trying to cluster mentions which came from different dir-halves")
                    exit(1)

                key1 = dir_num1 + extension1
                key2 = dir_num2 + extension2
                print("key:",key1)
                if dm1 not in dirHalfToDMs[doc_id]:
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
                #print("-----------\ncurrent doc:",str(doc_id),"\n-----------")
                
                # ensures we have all DMs
                if len(docToDMs[doc_id]) != len(self.corpus.docToDMs[doc_id]):
                    print("mismatch in DMs!!")
                    exit(1)

                # construct the golden truth for the current doc
                goldenTruthDirClusters = {}
                for i in range(len(self.corpus.docToREFs[doc_id])):
                    tmp = set()
                    curREF = self.corpus.docToREFs[doc_id][i]
                    for dm in self.corpus.docREFsToDMs[(doc_id,curREF)]:
                        # TMP:
                        if False: #self.args.runOnValid:
                            if dm not in self.helper.validDMs:
                                print("skipping:",str(dm))
                                continue
                        
                        tmp.add(dm)
                    goldenTruthDirClusters[i] = tmp
                    goldenSuperSet[goldenClusterID] = tmp
                    goldenClusterID += 1
                #print("golden clusters:", str(goldenTruthDirClusters))
                
                goldenK = len(self.corpus.docToREFs[doc_id])
                #print("# golden clusters: ",str(goldenK))
                # constructs our base clusters (singletons)
                ourDocClusters = {}
                for i in range(len(docToDMs[doc_id])):
                    dm = docToDMs[doc_id][i]
                    if False: #self.args.runOnValid:
                        if dm not in self.helper.validDMs:
                            print("skipping:",str(dm))
                            continue
                    
                    a = set()
                    a.add(dm)
                    ourDocClusters[i] = a

                #print("golden:",str(goldenTruthDirClusters))
                # the following keeps merging until our shortest distance > stopping threshold,
                # or we have 1 cluster, whichever happens first
                if not self.calculateMax:
                    while len(ourDocClusters.keys()) > 1:
                        # find best merge
                        closestDist = 999999
                        closestClusterKeys = (-1,-1)

                        closestAvgDist = 999999
                        closestAvgClusterKeys = (-1,-1)

                        closestAvgAvgDist = 999999
                        closestAvgAvgClusterKeys = (-1,-1)

                        #print("ourDocClusters:",str(ourDocClusters.keys()))
                        # looks at all combinations of pairs
                        i = 0
                        for c1 in ourDocClusters.keys():
                            
                            #print("c1:",str(c1))
                            j = 0
                            for c2 in ourDocClusters.keys():
                                if j > i:
                                    avgavgdists = []
                                    for dm1 in ourDocClusters[c1]:
                                        avgdists = []
                                        for dm2 in ourDocClusters[c2]:
                                            dist = 99999
                                            if (dm1,dm2) in docToDMPredictions[doc_id]:
                                                dist = docToDMPredictions[doc_id][(dm1,dm2)]
                                                avgavgdists.append(dist)
                                                avgdists.append(dist)
                                            elif (dm2,dm1) in docToDMPredictions[doc_id]:
                                                dist = docToDMPredictions[doc_id][(dm2,dm1)]
                                                avgavgdists.append(dist)
                                                avgdists.append(dist)
                                            else:
                                                print("* error, why don't we have either dm1 or dm2 in doc_id")
                                                exit(1)
                                            if dist < closestDist:
                                                closestDist = dist
                                                closestClusterKeys = (c1,c2)  
                                        avgDist = float(sum(avgdists)) / float(len(avgdists))
                                        if avgDist < closestAvgDist:
                                            closestAvgDist = avgDist
                                            closestAvgClusterKeys = (c1,c2)
                                    avgavgDist = float(sum(avgavgdists)) / float(len(avgavgdists))
                                    if avgavgDist < closestAvgAvgDist:
                                        closestAvgAvgDist = avgavgDist
                                        closestAvgAvgClusterKeys = (c1,c2)
                                j += 1
                            i += 1

                                                #print("closestdist is now:",str(closestDist),"which is b/w:",str(closestClusterKeys))
                            #print("trying to merge:",str(closestClusterKeys))

                        # only merge clusters if it's less than our threshold
                        #if closestDist > stoppingPoint:
                        # changed
                        if self.args.clusterMethod == "min" and closestDist > stoppingPoint:
                            break
                        elif self.args.clusterMethod == "avg" and closestAvgDist > stoppingPoint:
                            break
                        elif self.args.clusterMethod == "avgavg" and closestAvgAvgDist > stoppingPoint:
                            break

                        newCluster = set()
                        (c1,c2) = closestClusterKeys
                        if self.args.clusterMethod == "avg":
                            (c1,c2) = closestAvgClusterKeys
                        elif self.args.clusterMethod == "avgavg":
                            (c1,c2) = closestAvgAvgClusterKeys

                        for _ in ourDocClusters[c1]:
                            newCluster.add(_)
                        for _ in ourDocClusters[c2]:
                            newCluster.add(_)
                        ourDocClusters.pop(c1, None)
                        ourDocClusters.pop(c2, None)
                        ourDocClusters[c1] = newCluster
                    # end of current doc
                    for i in ourDocClusters.keys():
                        ourClusterSuperSet[ourClusterID] = ourDocClusters[i]
                        #print("setting ourClusterSuperSet[",str(ourClusterID),"] to:",str(ourDocClusters[i]))
                        ourClusterID += 1

            # end of going through every doc
            print("# total golden clusters:",str(len(goldenSuperSet.keys())))
            print("# total our clusters:",str(len(ourClusterSuperSet)))
            #print("stoppingPoints: ",str(stoppingPoints))
            #print("avg stopping point: ",str(float(sum(stoppingPoints))/float(len(stoppingPoints))))

            #self.writeCoNLLPerlFile("ourKeys.response",ourClusterSuperSet)
            #self.writeCoNLLPerlFile("ourGolden.keys",goldenSuperSet)      
            #print("finished writing")
            return (ourClusterSuperSet, goldenSuperSet)
    def analyzeResults(self, pairs, predictions, predictedClusters):

        # sanity check: ensures all pairs are accounted for
        predictedHMIDs = set()
        for p in pairs:
            (hm_id1,hm_id2) = p
            predictedHMIDs.add(hm_id1)
            predictedHMIDs.add(hm_id2)
        
        parsedHMIDs = set()
        numMissing = 0
        for doc_id in self.hddcrp_parsed.docToHMentions.keys():
            for hm in self.hddcrp_parsed.docToHMentions[doc_id]:
                parsedHMIDs.add(hm.hm_id)
                if hm.hm_id not in predictedHMIDs:
                    numMissing += 1
        if numMissing > 0:
            print("* ERROR: numMissing > 0")
            exit(1)
        print("predictedHMIDs:",str(len(predictedHMIDs)))
        print("parsedHMIDs:",str(len(parsedHMIDs)))
        print("# from parsing that we didnt' cluster:",str(numMissing))
        numMissing = 0
        for hm_id in predictedHMIDs:
            if hm_id not in parsedHMIDs:
                numMissing += 1
        print("# from predicting that we didn't parse:",str(numMissing))
        if numMissing > 0:
            exit(1)
        # end of sanity chk

        # stores distances from every hmention
        # hm_id1 -> {hm_id2 -> score}
        hmidToPredictions = defaultdict(lambda : defaultdict(float))

        # stores distances for every doc]
        # doc_id -> {(hm_id1,hm_id2) -> score}
        docToPredictions = defaultdict(lambda : defaultdict(float))
        for i in range(len(pairs)):
            (hm_id1,hm_id2) = pairs[i]
            pred = predictions[i][0]
            hmidToPredictions[hm_id1][hm_id2] = pred
            hmidToPredictions[hm_id2][hm_id1] = pred

            # sanity chk: ensures both hms belong to the same doc
            doc1 = self.hddcrp_parsed.hm_idToHMention[hm_id1].doc_id
            doc2 = self.hddcrp_parsed.hm_idToHMention[hm_id2].doc_id
            if doc1 != doc2:
                print("*ERROR: hms belong to diff docs")
                exit(1)

            if (hm_id1,hm_id2) not in docToPredictions[doc1] and (hm_id2,hm_id1) not in docToPredictions[doc1]:
                docToPredictions[doc1][(hm_id1,hm_id2)] = pred

        docToClusterIDs = defaultdict(set)

        # (1) sets hm_id to cluster num
        # (2) sets the predicted cluster IDs for each doc
        hm_idToPredictedClusterID = {}
        for c_id in predictedClusters.keys():
            for hm_id in predictedClusters[c_id]:
                hm_idToPredictedClusterID[hm_id] = c_id
                doc_id = self.hddcrp_parsed.hm_idToHMention[hm_id].doc_id
                docToClusterIDs[doc_id].add(c_id)

        # sets hm_id to golden ref cluster num on a per-doc basis
        # doc_id -> {REF -> hm_id}
        docToGoldenREF = defaultdict(lambda : defaultdict(set))
        for hm_id in hm_idToPredictedClusterID:
            doc_id = self.hddcrp_parsed.hm_idToHMention[hm_id].doc_id
            ref_id = self.hddcrp_parsed.hm_idToHMention[hm_id].ref_id
            docToGoldenREF[doc_id][ref_id].add(hm_id)

        # goes through each doc
        fout1 = open(str(self.args.resultsDir) + "tmp_goldenClusters.txt",'w')
        fout2 = open(str(self.args.resultsDir) + "tmp_preds.txt", "w")
        fout3 = open(str(self.args.resultsDir) + "tmp_allpreds.txt", "w")
        fout4 = open(str(self.args.resultsDir) + "tmp_predClusters.txt", "w")
        

        # computes accuracy
        docToF1 = {}
        docToRecall = {}
        docToPrec = {}
        for doc_id in self.hddcrp_parsed.docToHMentions.keys():
            num_correct = 0
            num_pred = 0
            num_golds = 0
            for hm in self.hddcrp_parsed.docToHMentions[doc_id]:
                hm_id1 = hm.hm_id
                sorted_distances = sorted(hmidToPredictions[hm_id1].items(), key=operator.itemgetter(1), reverse=False)
                gold_ref1 = hm.ref_id
                pred_ref1 = hm_idToPredictedClusterID[hm_id1]
                for (hm_id2,pred) in sorted_distances:
                    gold_ref2 = self.hddcrp_parsed.hm_idToHMention[hm_id2].ref_id
                    pred_ref2 = hm_idToPredictedClusterID[hm_id2]
                    if pred_ref1 == pred_ref2:
                        num_pred += 1
                        if gold_ref1 == gold_ref2: # we correctly got it
                            num_correct += 1
                    if gold_ref1 == gold_ref2:
                        num_golds += 1
            recall = 1
            if num_golds > 0:
                recall = float(num_correct) / float(num_golds)
            prec = 0
            if num_pred > 0:
                prec = float(num_correct) / float(num_pred)
            denom = float(recall + prec)
            docToRecall[doc_id] = recall
            docToPrec[doc_id] = prec
            f1 = 0
            if denom > 0:
                f1 = 2*(recall*prec) / float(denom)
            docToF1[doc_id] = f1

        dmPairsAnalyzed = set() # used for making the prec/recall tables to ensure we don't double count mentions
        tokenTables = defaultdict(lambda : defaultdict(int))
        # prints in order of best performing to worst (per pairwise accuracy)
        for (doc_id,f1) in sorted(docToF1.items(), key=operator.itemgetter(1), reverse=True):
            numREFs = len(docToGoldenREF[doc_id])
            fout1.write("\nDOC:" + str(doc_id) + " f1:" + str(f1) + " rec:" + str(docToRecall[doc_id]) + "; prec:" + str(docToPrec[doc_id]) + " [# REFS:" + str(numREFs) + "]\n---------------------\n")
            fout2.write("\nDOC:" + str(doc_id) + " f1:" + str(f1) + " rec:" + str(docToRecall[doc_id]) + "; prec:" + str(docToPrec[doc_id]) + " [# REFS:" + str(numREFs) + "]\n---------------------\n")
            fout3.write("\nDOC:" + str(doc_id) + " f1:" + str(f1) + " rec:" + str(docToRecall[doc_id]) + "; prec:" + str(docToPrec[doc_id]) + " [# REFS:" + str(numREFs) + "]\n---------------------\n")
            fout4.write("\nDOC:" + str(doc_id) + " f1:" + str(f1) + " rec:" + str(docToRecall[doc_id]) + "; prec:" + str(docToPrec[doc_id]) + " [# REFS:" + str(numREFs) + "]\n---------------------\n")
            for (pair,pred) in sorted(docToPredictions[doc_id].items(), key=operator.itemgetter(1), reverse=False):
                (hm_id1,hm_id2) = pair
                hmention1 = self.hddcrp_parsed.hm_idToHMention[hm_id1]
                hmention2 = self.hddcrp_parsed.hm_idToHMention[hm_id2]
                prefix = "  "
                tableKey = ""
                if len(hmention1.tokens) > 1 or len(hmention2.tokens) > 1:
                    prefix += "@"
                    tableKey = "M" # represents multiple tokens
                else: # both are single-token mentions
                    tableKey = "S"

                lemma1 = ""
                lemma1Tokens = []
                for htoken in hmention1.tokens:
                    token = self.corpus.UIDToToken[htoken.UID]
                    curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                    lemma1Tokens.append(curLemma)
                    lemma1 += curLemma + " "

                lemma2 = ""
                lemma2Tokens = []
                for htoken in hmention2.tokens:
                    token = self.corpus.UIDToToken[htoken.UID]
                    curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                    lemma2Tokens.append(curLemma)
                    lemma2 += curLemma + " "

                containsSubString = False
                for t in lemma1Tokens:
                    if t in lemma2:
                        containsSubString = True
                        break
                for t in lemma2Tokens:
                    if t in lemma1:
                        containsSubString = True
                        break
                if containsSubString:
                    tableKey += "SL"
                else:
                    tableKey += "NL"

                gold_ref1 = hmention1.ref_id
                pred_ref1 = hm_idToPredictedClusterID[hm_id1]
                gold_ref2 = hmention2.ref_id
                pred_ref2 = hm_idToPredictedClusterID[hm_id2]


                ans = ""
                if gold_ref1 == gold_ref2 and pred_ref1 == pred_ref2: # we got it
                    prefix += "**"
                    ans = "TP"
                elif gold_ref1 == gold_ref2 and pred_ref1 != pred_ref2: # we missed it
                    prefix += "-"
                    ans = "FN"
                elif gold_ref1 != gold_ref2 and pred_ref1 == pred_ref2: # false positive
                    prefix += "+"
                    ans = "FP"
                else:
                    ans = "TN"

                if (hm_id1,hm_id2) not in dmPairsAnalyzed and (hm_id2,hm_id1) not in dmPairsAnalyzed:
                    tokenTables[tableKey][ans] += 1

                    dmPairsAnalyzed.add((hm_id1,hm_id2))
                    dmPairsAnalyzed.add((hm_id2,hm_id1))
                fout3.write(str(prefix) + " " + str(hm_id1) + " (" + str(hmention1.getMentionText()) + " lem:" + str(lemma1) + ") and " + str(hm_id2) + " (" + str(hmention2.getMentionText()) + " lem:" + str(lemma2) + ") = " + str(pred) + "\n")

            for c_id in docToClusterIDs[doc_id]:
                fout4.write("\tCLUSTER:" + str(c_id) + "\n")
                for hm_id in predictedClusters[c_id]:
                    hmention = self.hddcrp_parsed.hm_idToHMention[hm_id]
                    fout4.write("\t\t[" + str(hm_id) + "]:" + str(hmention.getMentionText()) + "\n")

            for ref_id in docToGoldenREF[doc_id]:
                fout1.write("\tREF:" + str(ref_id) + "\n")
                for hm_id in docToGoldenREF[doc_id][ref_id]:
                    hmention = self.hddcrp_parsed.hm_idToHMention[hm_id]
                    fout1.write("\t\t[" + str(hm_id) + "]:" + str(hmention.getMentionText()) + "\n")

             # goes through each mention (redundanty, aka m1 -> all.. and m2 -> all)
            for hm in self.hddcrp_parsed.docToHMentions[doc_id]:
                hm_id1 = hm.hm_id
                sorted_distances = sorted(hmidToPredictions[hm_id1].items(), key=operator.itemgetter(1), reverse=False)
                gold_ref1 = hm.ref_id
                pred_ref1 = hm_idToPredictedClusterID[hm_id1]

                doc_id = hm.doc_id
                sentNum = hm.tokens[0].sentenceNum
                sent = ' '.join(self.hddcrp_parsed.docSentences[doc_id][sentNum])

                lemma1 = ""
                for htoken in hm.tokens:
                    token = self.corpus.UIDToToken[htoken.UID]
                    lemma1 += self.helper.getBestStanToken(token.stanTokens).lemma + " "
                    

                fout2.write("\nHMENTION:" + str(hm_id1) + " (" + str(hm.getMentionText()) + " lem:" + str(lemma1) + ") -- " + str(sent) + "\n")
                for (hm_id2,pred) in sorted_distances:
                    hmention2 = self.hddcrp_parsed.hm_idToHMention[hm_id2]

                    lemma2 = ""
                    for htoken in hmention2.tokens:
                        token = self.corpus.UIDToToken[htoken.UID]
                        lemma2 += self.helper.getBestStanToken(token.stanTokens).lemma + " "

                    gold_ref2 = self.hddcrp_parsed.hm_idToHMention[hm_id2].ref_id
                    pred_ref2 = hm_idToPredictedClusterID[hm_id2]
                    prefix = "  "
                    if gold_ref1 == gold_ref2 and pred_ref1 == pred_ref2: # we got it
                        prefix += "**"
                    elif gold_ref1 == gold_ref2 and pred_ref1 != pred_ref2: # we missed it
                        prefix += "-"
                    elif gold_ref1 != gold_ref2 and pred_ref1 == pred_ref2: # false positive
                        prefix += "+"

                    sent = ' '.join(self.hddcrp_parsed.docSentences[hmention2.doc_id][hmention2.tokens[0].sentenceNum])
                    fout2.write(str(prefix) + " " + str(hm_id2) + " (" + str(hmention2.getMentionText()) + " lem:" + str(lemma1) + ") = " + str(pred) + " -- " + str(sent) + "\n")
        
        for k in tokenTables:
            fout3.write("key:" + str(k) + ":" + str(tokenTables[k]) + "\n")
        fout1.close()
        fout2.close()
        fout3.close()
        fout4.close()
        '''
        for hm_id in hmidToPredictions:
            print("hm_id:",str(hm_id))
            sorted_distances = sorted(hmidToPredictions[hm_id].items(), key=operator.itemgetter(1), reverse=False)
            for s in sorted_distances:
                print("s:",str(s))
            exit(1)
        '''
        #

    def writePredictionsToFile(self, dev_pairs, dev_preds, testing_pairs, testing_preds):
        baseOut = str(self.args.resultsDir) + \
            str(self.args.hddcrpBaseFile) + "_" + \
            "nl" + str(self.args.numLayers) + "_" + \
            "pool" + str(self.args.poolType) + "_" + \
            "ne" + str(self.args.numEpochs) + "_" + \
            "ws" + str(self.args.windowSize) + "_" + \
            "neg" + str(self.args.numNegPerPos) + "_" + \
            "bs" + str(self.args.batchSize) + "_" + \
            "s" + str(self.args.shuffleTraining) + "_" + \
            "e" + str(self.args.embeddingsBaseFile) + "_" + \
            "dr" + str(self.args.dropout) + "_" + \
            "cm" + str(self.args.clusterMethod) + "_" + \
            "nf" + str(self.args.numFilters) + "_" + \
            "fm" + str(self.args.filterMultiplier) + "_" + \
            "fp" + str(self.args.featurePOS) + "_" + \
            "pt" + str(self.args.posType) + "_" + \
            "lt" + str(self.args.lemmaType) + "_" + \
            "dt" + str(self.args.dependencyType) + "_" + \
            "ct" + str(self.args.charType) + "_" + \
            "st" + str(self.args.SSType) + "_" + \
            "ws2" + str(self.args.SSwindowSize) + "_" + \
            "vs" + str(self.args.SSvectorSize) + "_" + \
            "sl" + str(self.args.SSlog) + "_" + \
            "dev" + str(self.args.devDir)
        foutdev = open(str(baseOut) + "_dev.txt", "w")
        fouttest = open(str(baseOut) + "_test.txt", "w")
        #foutdev = open("dev.txt", "w")
        #fouttest = open("test.txt", "w")
        # sanity check
        if len(dev_pairs) != len(dev_preds) or len(testing_pairs) != len(testing_preds):
            print("* ERROR: inconsistent sizes")
            exit(1)
        # end of sanity check

        for _ in range(len(dev_pairs)):
            ((d1,m1),(d2,m2)) = dev_pairs[_]
            foutdev.write(str(d1) + "," + str(m1) + "," + str(d2) + "," + str(m2) + "," + str(dev_preds[_][0]) + "\n")
        foutdev.close()
        foutdev.close()

        for _ in range(len(testing_pairs)):
            (hm1,hm2) = testing_pairs[_]
            fouttest.write(str(hm1) + "," + str(hm2) + "," + str(testing_preds[_][0]) + "\n")
        fouttest.close()

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

        # train
        if self.args.CCNNOpt == "rms":
            opt = RMSprop()
        elif self.args.CCNNOpt == "adam":
            opt = Adam()
        elif self.args.CCNNOpt == "adagrad":
            opt = Adagrad()
        else:
            print("* ERROR: invalid CCNN optimizer")
            exit(1)

        # loads embeddings for each word type
        self.loadEmbeddings(self.args.embeddingsFile, self.args.embeddingsType)
        print("# embeddings loaded:",str(len(self.wordTypeToEmbedding.keys())))
        # constructs the training and dev files
        training_pairs, training_data, training_rel, training_labels = self.createData("train", self.helper.trainingDirs)
        dev_pairs, dev_data, dev_rel, dev_labels = self.createData("dev", self.helper.devDirs)
        
        if self.args.useECBTest:
            testing_pairs, testing_data, testing_rel, testing_labels = self.createData("test", self.helper.testingDirs)
        else:
            testing_pairs, testing_data, testing_rel, testing_labels = self.createData("hddcrp")

        print("* training data shape:",str(training_data.shape))
        print("* dev data shape:",str(dev_data.shape))
        print("* test data shape:",str(testing_data.shape))
        print("# unique lemmas:",str(len(self.lemmas)))
        print("# of which were OOV:",str(len(self.OOVLemmas)))
        for _ in self.mentionLengthToMentions.keys():
            print("mentionLength:",str(_)," has ",str(len(self.mentionLengthToMentions[_])),"mentions")

        # network definition
        input_shape = training_data.shape[2:]
        base_network = self.create_base_network(input_shape)

        if self.useRelationalFeatures: # relational, merged layer way
            input_a = Input(shape=input_shape, name='input_a')
            input_b = Input(shape=input_shape, name='input_b')
        else: # original way
            input_a = Input(shape=input_shape)
            input_b = Input(shape=input_shape)
        
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([processed_a, processed_b])

        # relational, merged layer way
        if self.useRelationalFeatures:
            auxiliary_input = Input(shape=(len(training_rel[0]),), name='auxiliary_input')
            combined_layer = keras.layers.concatenate([distance, auxiliary_input])
            x = Dense(4, activation='relu')(combined_layer)
            main_output = Dense(1, activation='sigmoid', name='main_output')(x)
            model = Model([input_a, input_b, auxiliary_input], main_output)
            model.compile(loss=self.contrastive_loss, optimizer=opt)

        elif self.NNBasic: # the basic NN approach (instead of CCNN)
            model = Sequential()
            inputSize = len(training_data[0][0][0])
            model.add(Dense(units=600, input_shape=(inputSize,), use_bias=True, kernel_initializer='normal'))
            model.add(Activation('relu'))
            model.add(Dense(units=400, input_shape=(600,), use_bias=True, kernel_initializer='normal'))
            model.add(Activation('relu'))
            model.add(Dense(units=200, input_shape=(400,), use_bias=True, kernel_initializer='normal'))
            model.add(Activation('relu'))
            model.add(Dense(units=2, input_shape=(200,), use_bias=True, kernel_initializer='normal'))
            model.add(Activation('softmax'))
            model.compile(loss=self.weighted_binary_crossentropy,optimizer=opt,metrics=['accuracy'])

        else: # original
            model = Model(inputs=[input_a, input_b], outputs=distance)
            model.compile(loss=self.contrastive_loss, optimizer=opt)
            
        print(model.summary())

        if self.useRelationalFeatures: # relational, merged layer way
            model.fit({'input_a': np.asarray(training_data[:, 0]), 'input_b': np.asarray(training_data[:, 1]), 'auxiliary_input': np.asarray(training_rel)},
                      {'main_output': training_labels}, 
                      batch_size=self.args.batchSize,
                      epochs=self.args.numEpochs,
                      validation_data=({'input_a': np.asarray(dev_data[:, 0]), 'input_b': np.asarray(dev_data[:, 1]), 'auxiliary_input': np.asarray(dev_rel)}, {'main_output': np.asarray(dev_labels)}))
        elif self.NNBasic:
            (NNX, NNY) = self.transformToNNFormat(training_data, training_labels)
            (NNDEVX, NNDEVY) = self.transformToNNFormat(dev_data, dev_labels)
            model.fit(NNX, NNY, epochs=20, batch_size=self.args.batchSize, validation_data=(NNDEVX,NNDEVY), verbose=1)

        else: # original
            model.fit([training_data[:, 0], training_data[:, 1]], training_labels,
                      batch_size=self.args.batchSize,
                      epochs=self.args.numEpochs,
                      validation_data=([dev_data[:, 0], dev_data[:, 1]], dev_labels))
        # train accuracy
        print("-----------\npredicting training")
        
        if self.useRelationalFeatures:
            training_preds = model.predict({'input_a': np.asarray(training_data[:, 0]), 'input_b': np.asarray(training_data[:, 1]), 'auxiliary_input': np.asarray(training_rel)})
        
        elif self.NNBasic:
            (NNX, NNY) = self.transformToNNFormat(training_data, training_labels)
            training_preds = model.predict(NNX)
        else:
            training_preds = model.predict([training_data[:, 0], training_data[:, 1]])
            sys.stdout.flush()
            bestProb_train = self.compute_optimal_f1("training",0.5, training_preds, training_labels)
            print("training acc:", str(self.compute_accuracy(bestProb_train, training_preds, training_labels)))

        # dev accuracy
        print("-----------\npredicting dev")
        if self.useRelationalFeatures:
            dev_preds = model.predict({'input_a': np.asarray(dev_data[:, 0]), 'input_b': np.asarray(dev_data[:, 1]), 'auxiliary_input': np.asarray(dev_rel)})
        elif self.NNBasic:
            (NNDEVX, NNDEVY) = self.transformToNNFormat(dev_data, dev_labels)
            dev_preds = model.predict(NNDEVX)
        else:
            dev_preds = model.predict([dev_data[:, 0], dev_data[:, 1]])
            bestProb_dev = self.compute_optimal_f1("dev", bestProb_train, dev_preds, dev_labels)
            print("dev acc:", str(self.compute_accuracy(bestProb_dev, dev_preds, dev_labels)))
        
        # clears up ram
        training_pairs = None
        training_data = None
        training_labels = None
        dev_data = None
        dev_labels = None

        print("-----------\npredicting testing")

        if self.useRelationalFeatures:
            testing_preds = model.predict({'input_a': np.asarray(testing_data[:, 0]), 'input_b': np.asarray(testing_data[:, 1]), 'auxiliary_input': np.asarray(testing_rel)})
        elif self.NNBasic:
            (NNTESTX, NNTESTY) = self.transformToNNFormat(testing_data, testing_labels)
            testing_preds = model.predict(NNTESTX)
        else:
            testing_preds = model.predict([testing_data[:, 0], testing_data[:, 1]])
            bestProb_test = self.compute_optimal_f1("testing", bestProb_dev, testing_preds, testing_labels)
            print("test acc:", str(self.compute_accuracy(bestProb_test, testing_preds, testing_labels)))
            print("testing size:", str(len(testing_data)))

        #if not self.args.useECBTest:
        #    self.printSubstringTable(testing_pairs, testing_preds, bestProb_test)

        # TMP: remove this after i have tested if K-fold helps and is needed
        #self.writePredictionsToFile(dev_pairs, dev_preds, testing_pairs, testing_preds)

        return (dev_pairs, dev_preds, testing_pairs, testing_preds)
        
    # takes the CCNN format of pairs and labels (1 0 0 0 1) and transforms into format for our NN
    def transformToNNFormat(self, X, Y):
        newX = []
        newY = []
        for _ in range(len(X)):
            tmpX = []
            for eachDim in range(len(X[_][0][0])):
                tmpX.append(abs(X[_][0][0][eachDim][0] - X[_][1][0][eachDim][0]))
            newX.append(tmpX)
        for _ in Y:
            if _ == 1:
                newY.append([0,1])
            else:
                newY.append([1,0])
        return (newX, newY)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))
        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, 0.25)
        return K.mean(cost * 0.8, axis=-1)

    def loadPredictions(self, fileIn):
        testing_pairs = []
        testing_preds = []

        testingPredSum = defaultdict(float)

        numFiles = 23
        f = open(fileIn, "r")
        #fout = open("testAvg.txt", "w")
        for line in f:
            hm1,hm2,pred = line.rstrip().split(",")
            testingPredSum[(int(hm1),int(hm2))] += float(pred)
        f.close()

        fout = open("testavg.txt", "w")
        for (hm1,hm2) in testingPredSum:
            avgPred = float(testingPredSum[(hm1,hm2)] / 23.0)
            testing_pairs.append((hm1,hm2))
            testing_preds.append([avgPred])
            fout.write(str(hm1) + "," + str(hm2) + "," + str(avgPred) + "\n")
        fout.close()
        return (testing_pairs, testing_preds)

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
        curNumFilters = self.args.numFilters
        kernel_rows = 1
        if self.args.windowSize > 0:
            kernel_rows = 3

        seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
        seq.add(Dropout(float(self.args.dropout)))

        curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))
        seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
        curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))

        if kernel_rows == 3:
            kernel_rows = 2

        if self.args.poolType == "avg":
            seq.add(AveragePooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
        elif self.args.poolType == "max":
            seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
        else:
            print("* ERROR: invalid poolType; must be 'avg' or 'max'")
        
        # added following
        if self.args.numLayers == 2:
            print("going deep!! 2 sections of convolution")

            seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
            seq.add(Dropout(float(self.args.dropout)))

            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))
            seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))

            if self.args.poolType == "avg":
                seq.add(AveragePooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            elif self.args.poolType == "max":
                seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            else:
                print("* ERROR: invalid poolType; must be 'avg' or 'max'")
            
            # end of added
        elif self.args.numLayers == 3:
            print("going deeper!! 3 sections of convolution")
            seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
            seq.add(Dropout(float(self.args.dropout)))

            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))
            seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))

            if self.args.poolType == "avg":
                seq.add(AveragePooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            elif self.args.poolType == "max":
                seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            else:
                print("* ERROR: invalid poolType; must be 'avg' or 'max'")

            seq.add(Dropout(float(self.args.dropout)))
        
            # entering level 3
            seq.add(Conv2D(self.args.numFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", input_shape=input_shape, data_format="channels_first"))
            seq.add(Dropout(float(self.args.dropout)))

            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))
            seq.add(Conv2D(curNumFilters, kernel_size=(kernel_rows, 3), activation='relu', padding="same", data_format="channels_first"))
            curNumFilters = int(round(curNumFilters*self.args.filterMultiplier))

            if self.args.poolType == "avg":
                seq.add(AveragePooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            elif self.args.poolType == "max":
                seq.add(MaxPooling2D(pool_size=(kernel_rows, 2), padding="same", data_format="channels_first"))
            else:
                print("* ERROR: invalid poolType; must be 'avg' or 'max'")
            seq.add(Dropout(float(self.args.dropout)))
        
        seq.add(Flatten())
        seq.add(Dense(curNumFilters, activation='relu'))
        return seq

    # prints the stats for hte table which tells us how many T,F pairs we had
    # wrt to single-tokens and multi-token and containing sublemmas
    def printSubstringTable(self, testing_pairs, testing_preds, threshold):
        print("threshold:",str(threshold))
        dmPairsAnalyzed = set() # used for making the prec/recall tables to ensure we don't double count mentions
        tokenTables = defaultdict(lambda : defaultdict(int))        
        for _ in range(len(testing_pairs)):
            pair = testing_pairs[_]
            (hm_id1,hm_id2) = pair
            
            hmention1 = self.hddcrp_parsed.hm_idToHMention[hm_id1]
            hmention2 = self.hddcrp_parsed.hm_idToHMention[hm_id2]
            prefix = "  "
            tableKey = ""
            if len(hmention1.tokens) > 1 or len(hmention2.tokens) > 1:
                tableKey = "M" # represents multiple tokens
            else: # both are single-token mentions
                tableKey = "S"

            lemma1 = ""
            lemma1Tokens = []
            for htoken in hmention1.tokens:
                token = self.corpus.UIDToToken[htoken.UID]
                curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                lemma1Tokens.append(curLemma)
                lemma1 += curLemma + " "

            lemma2 = ""
            lemma2Tokens = []
            for htoken in hmention2.tokens:
                token = self.corpus.UIDToToken[htoken.UID]
                curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                lemma2Tokens.append(curLemma)
                lemma2 += curLemma + " "

            containsSubString = False
            for t in lemma1Tokens:
                if t in lemma2:
                    containsSubString = True
                    break
            for t in lemma2Tokens:
                if t in lemma1:
                    containsSubString = True
                    break
            if containsSubString:
                tableKey += "SL"
            else:
                tableKey += "NL"

            gold_ref1 = hmention1.ref_id
            gold_ref2 = hmention2.ref_id

            wePredict = False
            pred = testing_preds[_]
            if pred < threshold:
                wePredict = True

            ans = ""
            if gold_ref1 == gold_ref2 and wePredict: # we got it
                ans = "TP"
            elif gold_ref1 == gold_ref2 and not wePredict: # we missed it
                ans = "FN"
            elif gold_ref1 != gold_ref2 and wePredict: # false positive
                ans = "FP"
            else:
                ans = "TN"

            if (hm_id1,hm_id2) not in dmPairsAnalyzed and (hm_id2,hm_id1) not in dmPairsAnalyzed:
                tokenTables[tableKey][ans] += 1
        for tableKey in tokenTables:
            for ans in tokenTables[tableKey]:
                print("tokenTables:",str(tableKey),str(ans),str(tokenTables[tableKey][ans]))

    # from a list of predictions, find the optimal f1 point
    def compute_optimal_f1(self, label, startingProb, predictions, golds):
        #print("* in compute_optimal_f1!!!()")
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
        
        lowestProb = 0.1
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

        self.helper.wordEmbLength = self.embeddingLength # lemmas use this

        self.wordTypeToEmbedding["'knows"] = self.wordTypeToEmbedding["knows"]
        self.wordTypeToEmbedding["takeing"] = self.wordTypeToEmbedding["taking"]
        self.wordTypeToEmbedding["arested"] = self.wordTypeToEmbedding["arrested"]
        self.wordTypeToEmbedding["arest"] = self.wordTypeToEmbedding["arrest"]
        self.wordTypeToEmbedding["intpo"] = self.wordTypeToEmbedding["into"]        
        self.wordTypeToEmbedding["texa"] = self.wordTypeToEmbedding["texas"]
        self.wordTypeToEmbedding["itune"] = self.wordTypeToEmbedding["itunes"]
        self.wordTypeToEmbedding["degenere"] = self.wordTypeToEmbedding["degeneres"]
        self.wordTypeToEmbedding["#oscars"] = self.wordTypeToEmbedding["oscars"]
        self.wordTypeToEmbedding["microserver"] = self.wordTypeToEmbedding["server"]
        self.wordTypeToEmbedding["microservers"] = self.wordTypeToEmbedding["servers"]
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

    def getSSEmbedding(self, SSType, tokenList):
        ssEmb = []
        if SSType == "none":
            return ssEmb
        elif SSType == "sum" or SSType == "avg":
            ssLength = self.helper.SSEmbLength
            sumEmb = [0]*ssLength
            numFound = 0
            for t in tokenList:
                if t.text in self.helper.SSMentionTypeToVec.keys():
                    curEmb = self.helper.SSMentionTypeToVec[t.text]
                    sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]
                    numFound += 1
                else:
                    print("* WARNING: we didn't find:",str(t.text),"in SSMentionTypeToVec")

            ssEmb = sumEmb
            if SSType == "avg" and numFound > 1:
                avgEmb = [x / float(numFound) for x in sumEmb]
                ssEmb = avgEmb
            return ssEmb
        else: # can't be none, since we've specified featurePOS
            print("* ERROR: SSType is illegal")
            exit(1)

    def getCharEmbedding(self, charType, tokenList):
        charEmb = []
        if charType == "none" or len(tokenList) == 0: # as opposed to sum or avg
            return charEmb
        elif charType == "sum" or charType == "avg":
            charLength = self.helper.charEmbLength

            # sum over all tokens first, optionally avg
            sumEmb = [0]*charLength
            numCharsFound = 0
            for t in tokenList:
                lemma = self.helper.getBestStanToken(t.stanTokens).lemma

                for char in lemma:
                    if char in self.helper.charToEmbedding.keys():
                        curEmb = self.helper.charToEmbedding[char]
                        sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]
                        numCharsFound += 1
                    else:
                        print("* WARNING: we don't have char:",str(char))

            if charType == "avg":
                if numCharsFound > 1:
                    charEmb = [x / float(numCharsFound) for x in sumEmb]
                else:
                    charEmb = sumEmb
                print("sum:",str(sumEmb))
                print("numCharsFound:",str(numCharsFound))
                print("avg:",str(charEmb))
            elif charType == "sum":
                charEmb = sumEmb

        elif charType == "concat":
            numCharsFound = 0
            for t in tokenList:
                lemma = self.helper.getBestStanToken(t.stanTokens).lemma
                for char in lemma:
                    if char in self.helper.charToEmbedding.keys():
                        if numCharsFound == 20:
                            break
                        else:
                            curEmb = self.helper.charToEmbedding[char]
                            charEmb += curEmb
                            numCharsFound += 1
                    else:
                        print("* ERROR: we don't have char:",str(char))
                        exit(1)

            while len(charEmb) < 20*self.helper.charEmbLength:
                charEmb.append(0.0)

        else: # can't be none, since we've specified featurePOS
            print("* ERROR: charType is illegal")
        return charEmb

    def getDependencyEmbedding(self, dependencyType, tokenList):
        dependencyEmb = []
        if dependencyType == "none": # as opposed to sum or avg
            return dependencyEmb
        elif dependencyType == "sum" or dependencyType == "avg":

            # sum over all tokens first, optionally avg
            sumParentEmb = [0]*self.embeddingLength
            sumChildrenEmb = [0]*self.embeddingLength

            numParentFound = 0
            tmpParentLemmas = []
            numChildrenFound = 0
            tmpChildrenLemmas = []
            for t in tokenList:
                bestStanToken = self.helper.getBestStanToken(t.stanTokens)
                
                if len(bestStanToken.parentLinks) == 0:
                    print("* token has no dependency parent!")
                    exit(1)
                for stanParentLink in bestStanToken.parentLinks:
                    parentLemma = self.helper.removeQuotes(stanParentLink.parent.lemma)
                    curEmb = [0]*self.embeddingLength
                    
                    # TMP: just to see which texts we are missing
                    tmpParentLemmas.append(parentLemma)

                    if parentLemma == "ROOT":
                        curEmb = [1]*self.embeddingLength
                    else:
                        curEmb = self.wordTypeToEmbedding[constructECBTrainingparentLemma]
                    
                    isOOV = True
                    for _ in curEmb:
                        if _ != 0:
                            isOOV = False
                            numParentFound += 1
                    
                    sumParentEmb = [x + y for x,y in zip(sumParentEmb, curEmb)]
                
                # makes embedding for the dependency children's lemmas
                if len(bestStanToken.childLinks) == 0:
                    print("* token has no dependency children!")
                for stanChildLink in bestStanToken.childLinks:
                    childLemma = self.helper.removeQuotes(stanChildLink.child.lemma)
                    curEmb = [0]*self.embeddingLength
                    
                    # TMP: just to see which texts we are missing
                    tmpChildrenLemmas.append(childLemma)

                    if childLemma == "ROOT":
                        curEmb = [1]*self.embeddingLength
                    else:
                        curEmb = self.wordTypeToEmbedding[childLemma]
                    
                    isOOV = True
                    for _ in curEmb:
                        if _ != 0:
                            isOOV = False
                            numChildrenFound += 1
                    
                    sumChildrenEmb = [x + y for x,y in zip(sumChildrenEmb, curEmb)]
                
            # makes parent emb
            parentEmb = sumParentEmb
            if numParentFound == 0:
                print("* WARNING: numParentFound 0:",str(tmpParentLemmas))       
            if dependencyType == "avg" and numParentFound > 1:
                parentEmb = [x / float(numParentFound) for x in sumParentEmb]

            # makes chid emb
            childrenEmb = sumChildrenEmb
            if numChildrenFound == 0:
                print("* WARNING: numChildrenFound 0:",str(tmpChildrenLemmas))       
            if dependencyType == "avg" and numChildrenFound > 1:
                childrenEmb = [x / float(numChildrenFound) for x in sumChildrenEmb]

            return parentEmb + childrenEmb
        else: # can't be none, since we've specified featurePOS
            print("* ERROR: dependencyType is illegal")

    def getLemmaEmbedding(self, lemmaType, tokenList):
        lemmaEmb = []
        if lemmaType == "none": # as opposed to sum or avg
            return lemmaEmb
        elif lemmaType == "sum" or lemmaType == "avg":
            lemmaLength = self.helper.wordEmbLength

            # sum over all tokens first, optionally avg
            sumEmb = [0]*lemmaLength
            for t in tokenList:
                lemma = self.helper.getBestStanToken(t.stanTokens).lemma
                curEmb = self.wordTypeToEmbedding[lemma]
                sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]

                # TMP: sanity chk
                isEmpty = True
                for _ in curEmb:
                    if _ != 0:
                        isEmpty = False
                        break
                if isEmpty:
                    print("** WARNING: we didn't have legit emb for lemma:",str(lemma))

            if lemmaType == "avg":
                avgEmb = [x / float(len(tokenList)) for x in sumEmb]
                lemmaEmb = avgEmb
            elif lemmaType == "sum":
                lemmaEmb = sumEmb
                #print("lemmaEmb:",str(lemmaEmb))
        else: # can't be none, since we've specified featurePOS
            print("* ERROR: lemmaType is illegal")
        return lemmaEmb

    def getPOSEmbedding(self, featurePOS, posType, tokenList):
        posEmb = []
        if featurePOS == "none":
            return posEmb
        elif featurePOS == "onehot" or featurePOS == "emb_random" or featurePOS == "emb_glove":
            posLength = 50

            if featurePOS == "emb_random" or featurePOS == "emb_glove":
                posLength = self.helper.posEmbLength

            # sum over all tokens first, optionally avg
            if posType == "sum" or posType == "avg":
                sumEmb = [0]*posLength

                for t in tokenList:
                    # our current 1 ECB Token possibly maps to multiple StanTokens, so let's
                    # ignore the StanTokens that are ‘’ `` POS $, if possible (they may be our only ones)
                    pos = ""
                    posOfLongestToken = ""
                    longestToken = ""
                    for stanToken in t.stanTokens:
                        if stanToken.pos in self.helper.badPOS:
                            # only use the badPOS if no others have been set
                            if pos == "":
                                pos = stanToken.pos
                        else: # save the longest, nonBad POS tag
                            if len(stanToken.text) > len(longestToken):
                                longestToken = stanToken.text
                                posOfLongestToken = stanToken.pos 

                    if posOfLongestToken != "":
                        pos = posOfLongestToken
                    if pos == "":
                        print("* ERROR: our POS empty!")
                        exit(1)

                    curEmb = [0]*posLength
                    if featurePOS == "onehot":
                        curEmb[self.helper.posToIndex[pos]] += 1
                        
                    elif featurePOS == "emb_random":
                        curEmb = self.helper.posToRandomEmbedding[pos]
                    elif featurePOS == "emb_glove":
                        curEmb = self.helper.posToGloveEmbedding[pos]
                    sumEmb = [x + y for x,y in zip(sumEmb, curEmb)]

                if posType == "avg":
                    avgEmb = [x / float(len(tokenList)) for x in sumEmb]
                    posEmb = avgEmb
                elif posType == "sum":
                    posEmb = sumEmb

                #print("posEmb:",str(posEmb))
            else: # can't be none, since we've specified featurePOS
                print("* ERROR: posType is illegal")
        return posEmb


    # creates data from ECBCorpus (train and dev uses this, and optionally test)
    def createData(self, subset, dirs=None):

        if subset == "train":
            (tokenListPairs, mentionIDPairs, labels) = self.helper.constructECBTraining(dirs, self.isWDModel)
        elif subset == "dev":
            (tokenListPairs, mentionIDPairs, labels) = self.helper.constructECBDev(dirs, False, self.isWDModel)
        elif subset == "test":
            # this is not a mistake; constructECBDev() merely fetches all examples (no negative-subsampling),
            # so it's okay to re-use it to get the testing data
            (tokenListPairs, mentionIDPairs, labels) = self.helper.constructECBDev(dirs, True, self.isWDModel)
        elif subset == "hddcrp":
            (tokenListPairs, mentionIDPairs, labels) = self.helper.constructHDDCRPTest(self.hddcrp_parsed, self.isWDModel) # could be gold test or predicted test mentions
        else:
            print("* ERROR: unknown passed-in 'subset' param")
            exit(1)

        # lists can't be dictionary keys, so let's create a silly, temp mapping,
        # which will only be used in this function
        mentionIDToTokenList = {}
        for i in range(len(mentionIDPairs)):
            (mentionID1,mentionID2) = mentionIDPairs[i]
            (tokenList1,tokenList2) = tokenListPairs[i]
            mentionIDToTokenList[mentionID1] = tokenList1
            mentionIDToTokenList[mentionID2] = tokenList2

        # determines which mentions we'll construct
        mentionIDsWeCareAbout = set()
        for (mentionID1,mentionID2) in mentionIDPairs:
            mentionIDsWeCareAbout.add(mentionID1)
            mentionIDsWeCareAbout.add(mentionID2)

        # constructs the tokenList matrix for every mention
        mentionIDToMatrix = {}

        numRows = 1 #1 + 2*self.args.windowSize
        numCols = self.embeddingLength

        # TMP: preprocesses -- calculates frequency counts per mention tokens' lemma
        '''
        for mentionID in mentionIDsWeCareAbout:
            tokenList = mentionIDToTokenList[mentionID]
            for _ in tokenList:
                curLemma = self.helper.getBestStanToken(_.stanTokens, _).lemma
                curDir = _.doc_id[0:_.doc_id.find("_")]
                curDoc = _.doc_id
                self.mentionLemmaTokenCounts[curLemma] +=1
                self.mentionLemmaTokenDirCounts[curDir][curLemma] += 1
                self.mentionLemmaTokenDocCounts[curDoc][curLemma] += 1
        for doc_id in self.mentionLemmaTokenDocCounts:
            print("doc_id:",str(doc_id))
            for l in self.mentionLemmaTokenDocCounts[doc_id]:
                print("lemma:",str(l),self.mentionLemmaTokenDocCounts[doc_id][l])
        '''
        for mentionID in mentionIDsWeCareAbout:

            tokenList = mentionIDToTokenList[mentionID]

            # just for understanding our data more
            self.mentionLengthToMentions[len(tokenList)].append(tokenList)

            t_startIndex = 99999999
            t_endIndex = -1

            # gets token indices and constructs the Mention embedding
            sumGloveEmbedding = [0]*self.embeddingLength
            numTokensFound = 0

            for token in tokenList:

                cleanedStan = self.helper.removeQuotes(self.helper.getBestStanToken(token.stanTokens).text)
                cleanedText = self.helper.removeQuotes(token.text)

                if cleanedText in self.wordTypeToEmbedding.keys():
                    curEmbedding = self.wordTypeToEmbedding[cleanedText]
                else:
                    curEmbedding = self.wordTypeToEmbedding[cleanedStan]
                hasEmbedding = False
                for _ in curEmbedding:
                    if _ != 0:
                        hasEmbedding = True
                        break

                if hasEmbedding:
                    numTokensFound += 1
                    sumGloveEmbedding = [x + y for x,y in zip(sumGloveEmbedding, curEmbedding)]
                #print("curEmbedding:",str(curEmbedding))
                ind = self.corpus.corpusTokensToCorpusIndex[token]
                if ind < t_startIndex:
                    t_startIndex = ind
                if ind > t_endIndex:
                    t_endIndex = ind

            if numTokensFound > 0:
                avgGloveEmbedding = [x / float(numTokensFound) for x in sumGloveEmbedding]
            else:
                avgGloveEmbedding = sumGloveEmbedding
                print("* WARNING: we had 0 tokens of:",str(tokenList))
                for t in tokenList:
                    print("t:",str(t.text))
            # load other features
            posEmb = self.getPOSEmbedding(self.args.featurePOS, self.args.posType, tokenList)
            lemmaEmb = self.getLemmaEmbedding(self.args.lemmaType, tokenList)
            dependencyEmb = self.getDependencyEmbedding(self.args.dependencyType, tokenList)
            charEmb = self.getCharEmbedding(self.args.charType, tokenList)
            ssEmb = self.getSSEmbedding(self.args.SSType, tokenList)
            fullMenEmbedding = posEmb + lemmaEmb + dependencyEmb + charEmb + ssEmb #avgGloveEmbedding
            #print("fullMenEmbedding:",str(fullMenEmbedding))

            # sets the center
            # BELOW IS THE PROPER, ORIGINAL WAY
            #curMentionMatrix = np.zeros(shape=(numRows,len(fullMenEmbedding)))
            #curMentionMatrix[self.args.windowSize] = fullMenEmbedding

            # the prev tokens
            tmpTokenList = []
            for i in range(self.args.windowSize):
                ind = t_startIndex - self.args.windowSize + i

                pGloveEmb = [0]*self.embeddingLength    
                if ind >= 0:
                    token = self.corpus.corpusTokens[ind]
                    cleanedStan = self.helper.removeQuotes(self.helper.getBestStanToken(token.stanTokens).text)
                    cleanedText = self.helper.removeQuotes(token.text)
                    tmpTokenList.append(token)
                    if cleanedText in self.wordTypeToEmbedding:
                        pGloveEmb = self.wordTypeToEmbedding[cleanedText]
                    else:
                        pGloveEmb = self.wordTypeToEmbedding[cleanedStan]
                        print("* WARNING, we don't have:",str(token.text))
                        #exit(1)
                #curMentionMatrix[i] = fullTokenEmbedding
            prevTokenEmbedding = []
            if len(tmpTokenList) > 0:
                prevPosEmb = self.getPOSEmbedding(self.args.featurePOS, self.args.posType, tmpTokenList)
                prevLemmaEmb = self.getLemmaEmbedding(self.args.lemmaType, tmpTokenList)
                prevDependencyEmb = self.getDependencyEmbedding(self.args.dependencyType, tmpTokenList)
                prevCharEmb = self.getCharEmbedding(self.args.charType, tmpTokenList)
                prevSSEmb = self.getSSEmbedding(self.args.SSType, tmpTokenList)
                prevTokenEmbedding = prevPosEmb + prevLemmaEmb + prevDependencyEmb + prevCharEmb + prevSSEmb #prevDependencyEmb #pGloveEmb + prevPosEmb + prevLemmaEmb # 

            # gets the 'next' tokens
            tmpTokenList = []
            for i in range(self.args.windowSize):
                ind = t_endIndex + 1 + i

                nGloveEmb = [0]*self.embeddingLength
                
                #original: tmpTokenList = []
                if ind < self.corpus.numCorpusTokens - 1:
                    token = self.corpus.corpusTokens[ind]
                    cleanedStan = self.helper.removeQuotes(self.helper.getBestStanToken(token.stanTokens).text)
                    cleanedText = self.helper.removeQuotes(token.text)
                    tmpTokenList.append(token)
                    if cleanedText in self.wordTypeToEmbedding:
                        nGloveEmb = self.wordTypeToEmbedding[cleanedText]
                    else:
                        nGloveEmb = self.wordTypeToEmbedding[cleanedStan]
                        print("* WARNING, we don't have:",str(token.text))
            nextTokenEmbedding = []
            if len(tmpTokenList) > 0:
                nextPosEmb = self.getPOSEmbedding(self.args.featurePOS, self.args.posType, tmpTokenList)
                nextLemmaEmb = self.getLemmaEmbedding(self.args.lemmaType, tmpTokenList)
                nextDependencyEmb = self.getDependencyEmbedding(self.args.dependencyType, tmpTokenList)
                nextCharEmb = self.getCharEmbedding(self.args.charType, tmpTokenList)
                nextSSEmb = self.getSSEmbedding(self.args.SSType, tmpTokenList)
                nextTokenEmbedding = nextPosEmb + nextLemmaEmb + nextDependencyEmb + nextCharEmb + nextSSEmb
                #sumNextTokenEmbedding = [x + y for x,y in zip(sumNextTokenEmbedding, nextTokenEmbedding)]
                #curMentionMatrix[self.args.windowSize+1+i] = fullTokenEmbedding

            # NEW
            fullEmbedding = prevTokenEmbedding + fullMenEmbedding + nextTokenEmbedding
            '''
            print("nextTokenEmbedding:",str(len(nextTokenEmbedding)))
            print("prevTokenEmbedding:",str(len(prevTokenEmbedding)))
            print("fullMenEmbedding:",str(len(fullMenEmbedding)))
            print("full:",str(len(fullEmbedding)))
            '''
            curMentionMatrix = np.zeros(shape=(1,len(fullEmbedding)))
            curMentionMatrix[0] = fullEmbedding    
            curMentionMatrix = np.asarray(curMentionMatrix).reshape(numRows,len(fullEmbedding),1)
            
            # old way
            #curMentionMatrix = np.asarray(curMentionMatrix).reshape(numRows,len(fullMenEmbedding),1)

            mentionIDToMatrix[mentionID] = curMentionMatrix

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
        relational_features = []
        for _ in range(len(mentionIDPairs)):
            (mentionID1,mentionID2) = mentionIDPairs[_]
            pair = np.asarray([mentionIDToMatrix[mentionID1],mentionIDToMatrix[mentionID2]])
            X.append(pair)

            # makes 'contains a shared lemma substring' feature
            mention1Texts = ""
            mention1Lemmas = ""
            lemma1Tokens = []
            for token in mentionIDToTokenList[mentionID1]:
                mention1Texts += token.text + " "
                curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                mention1Lemmas += curLemma + " "
                lemma1Tokens.append(curLemma)
            mention1Texts = mention1Texts.rstrip()

            mention2Texts = ""
            mention2Lemmas = ""
            lemma2Tokens = []
            for token in mentionIDToTokenList[mentionID2]:
                mention2Texts += token.text + " "
                curLemma = self.helper.getBestStanToken(token.stanTokens).lemma
                mention2Lemmas += curLemma + " "
                lemma2Tokens.append(curLemma)
            mention2Texts = mention2Texts.rstrip()

            containsSubString = False
            for t in lemma1Tokens:
                if t in mention2Lemmas:
                    containsSubString = True
                    break
            for t in lemma2Tokens:
                if t in mention1Lemmas:
                    containsSubString = True
                    break

            curRelational = []

            # feature 1
            '''
            if len(lemma1Tokens) == 1 and len(lemma1Tokens) == 1: # both are singletons
                curRelational.append(0)
            else: # at least one is multi-token
                curRelational.append(1)

            # feature 2
            if containsSubString:
                curRelational.append(1)
            else:
                curRelational.append(0)
            '''

            # feature 4
            sed = self.levenshtein(mention1Texts,mention2Texts)
            curRelational.append(sed)
            # features:
            # 1: both are single-token (0) or at least one mention is multi-token (1)
            # 2: is the lemma of any word in either mention a substring of the lemma concat of the other (0 or 1)
            # 4: string edit distance of the mentions' texts
            relational_features.append(np.asarray(curRelational))

        Y = np.asarray(labels)
        X = np.asarray(X)
        relational_features = np.asarray(relational_features)

        ''' ORIGINAL WAY WHICH WORKS
        X = []
        for (mentionID1,mentionID2) in mentionIDPairs:
            pair = np.asarray([mentionIDToMatrix[mentionID1],mentionIDToMatrix[mentionID2]])
            X.append(pair)
        Y = np.asarray(labels)
        X = np.asarray(X)
        '''
        return (mentionIDPairs, X, relational_features, Y)

    def levenshtein(self, seq1, seq2, max_dist=-1):
        if seq1 == seq2:
            return 0
        len1, len2 = len(seq1), len(seq2)
        if max_dist >= 0 and abs(len1 - len2) > max_dist:
            return -1
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
        if len1 < len2:
            len1, len2 = len2, len1
            seq1, seq2 = seq2, seq1
        
        column = array('L', range(len2 + 1))
        
        for x in range(1, len1 + 1):
            column[0] = x
            last = x - 1
            for y in range(1, len2 + 1):
                old = column[y]
                cost = int(seq1[x - 1] != seq2[y - 1])
                column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
                last = old
            if max_dist >= 0 and min(column) > max_dist:
                return -1
        
        if max_dist >= 0 and column[len2] > max_dist:
            # stay consistent, even if we have the exact distance
            return -1
        return column[len2]