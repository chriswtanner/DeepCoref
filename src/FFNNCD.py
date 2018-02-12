import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.utils import np_utils
from collections import defaultdict
from tensorflow.python.client import device_lib
import numpy as np
import tensorflow as tf
import random
import keras.backend as K
import os
import sys
import functools
import math
from itertools import product
class FFNNCD:
	def __init__(self, args, corpus, helper, hddcrp_parsed, dev_pairs=None, dev_preds=None, testing_pairs=None, testing_preds=None):

		self.ChoubeyFilter = False # if True, remove the False Positives.  Missed Mentions still exist though.

		# print stuff
		print("args:", str(args))
		print("tf version:",str(tf.__version__))
		if args.device == "cpu":
			sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
			os.environ['CUDA_VISIBLE_DEVICES'] = ''
			print("session:",str(sess))
		print("devices:",device_lib.list_local_devices())

		self.args = args
		self.corpus = corpus
		self.helper = helper
		self.hddcrp_parsed = hddcrp_parsed

		# to be filled in via createTraining()
		self.testX = ""
		self.testY = ""

		self.model = None # filled in via train()

		self.createTraining(dev_pairs, dev_preds) # loads training/test data

		self.testingPairs = testing_pairs
		self.testingPreds = testing_preds

		# params
		self.hidden_size = 50
		self.dataDim = len(self.trainX[0])
		self.outputDim = 2
		self.batch_size = 5

		# the passed-in params
		self.num_epochs = int(self.args.FFNNnumEpochs)
		self.FFNNOpt = self.args.FFNNOpt
		pos_ratio = float(self.args.FFNNPosRatio) # 0.8
		neg_ratio = 1. - pos_ratio
		self.pos_ratio = tf.constant(pos_ratio, tf.float32)
		self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)

	def train(self):
		self.model = Sequential()
		self.model.add(Dense(units=self.hidden_size, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		self.model.add(Activation('sigmoid'))
		self.model.add(Dense(units=self.outputDim, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		self.model.add(Activation('softmax'))

		if self.FFNNOpt == "rms":
			opt = RMSprop()
		elif self.FFNNOpt == "adam":
			opt = Adam(lr=0.001)
		elif self.FFNNOpt == "adagrad":
			opt = Adagrad(lr=0.001)
		else:
			print("* ERROR: invalid CCNN optimizer")
			exit(1)
		self.model.compile(loss=self.weighted_binary_crossentropy,optimizer=opt,metrics=['accuracy'])
		self.model.summary()
		self.model.fit(self.trainX, self.trainY, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)

	def calculateF12(self, preds):
		num_correct = 0
		num_pred = 0
		num_golds = 0

		for _ in range(len(preds)):
			if preds[_] > 0.5:
				num_pred += 1
				if self.testY[_][1] > 0.5:
					num_correct += 1
			if self.testY[_][1] > 0.5:
				num_golds += 1
		recall = 1
		if num_golds > 0:
			recall = float(num_correct) / float(num_golds)
		prec = 0
		if num_pred > 0:
			prec = float(num_correct) / float(num_pred)
		denom = float(recall + prec)
		f1 = 0
		if denom > 0:
			f1 = 2*(recall*prec) / float(denom)
		return f1

	def calculateF1(self, preds):
		num_correct = 0
		num_pred = 0
		num_golds = 0
		for _ in range(len(preds)):
			if preds[_][1] > 0.5:
				num_pred += 1
				if self.testY[_][1] > 0.5:
					num_correct += 1
			if self.testY[_][1] > 0.5:
				num_golds += 1
		recall = 1
		if num_golds > 0:
			recall = float(num_correct) / float(num_golds)
		prec = 0
		if num_pred > 0:
			prec = float(num_correct) / float(num_pred)
		denom = float(recall + prec)
		f1 = 0
		if denom > 0:
			f1 = 2*(recall*prec) / float(denom)
		return f1

	def weighted_binary_crossentropy(self, y_true, y_pred):
		# Transform to logits
		epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
		y_pred = tf.log(y_pred / (1 - y_pred))

		cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
		#return K.mean(cost, axis=-1)
		return K.mean(cost * self.pos_ratio, axis=-1)

	def createTraining(self, dev_pairs, dev_preds):
		print("in FFNNCD's CreateTraining()")

		parsedDevDMs = set()
		for d in self.helper.devDirs:
			for doc in self.corpus.dirToDocs[d]:
				for dm in self.corpus.docToDMs[doc]:
					parsedDevDMs.add(dm)

		# loads dev predictions via CCNN's pairwise predictions, or a saved file
		dirHalfToDMPredictions = defaultdict(lambda : defaultdict(float))
		dirHalfToDMs = defaultdict(list) # used for ensuring our predictions included ALL valid DMs
		predDevDMs = set() # only used for sanity check
		if dev_pairs == None and dev_preds == None: # read the file
			fn = self.args.dataDir + "dev_" + str(self.args.devDir) + ".txt"
			print("** READING THE FILE:",str(fn))
			print("** ERROR: I believe I got rid of this codeflow.  Maybe it's used for HDDCRP comparison?  Exiting")
			exit(1)
			f = open(fn)
			for line in f:
				d1,m1,d2,m2,pred = line.rstrip().split(",")
				m1 = int(m1)
				m2 = int(m2)
				dm1 = (d1,m1)
				dm2 = (d2,m2)
				pred = float(pred)
				predDevDMs.add(dm1)
				predDevDMs.add(dm2)
			f.close()		
		else: # read the CCNN predictions
			print("** LOOKING AT CCNN's PREDICTIONS")
			for i in range(len(dev_pairs)):
				(dm1,dm2) = dev_pairs[i]
				
				pred = dev_preds[i][0]
				doc_id1 = dm1[0]
				doc_id2 = dm2[0]
				
				extension1 = doc_id1[doc_id1.find("ecb"):]
				dir_num1 = int(doc_id1.split("_")[0])
				
				extension2 = doc_id2[doc_id2.find("ecb"):]
				dir_num2 = int(doc_id2.split("_")[0])
				
				dirHalf1 = str(dir_num1) + extension1
				dirHalf2 = str(dir_num2) + extension2
				
				if dirHalf1 != dirHalf2:
					print("* ERROR, somehow, training pairs came from diff dir-halves")
					exit(1)

				if dm1 not in dirHalfToDMs[dirHalf1]:
					dirHalfToDMs[dirHalf1].append(dm1)
				if dm2 not in dirHalfToDMs[dirHalf2]:
					dirHalfToDMs[dirHalf2].append(dm2)

				dirHalfToDMPredictions[dirHalf1][(dm1,dm2)] = pred
				predDevDMs.add(dm1)
				predDevDMs.add(dm2)

		# sanity check part 1: ensures we have parsed all of our predicted DMs
		docToPredDevDMs = defaultdict(set)
		for dm in predDevDMs:
			if dm not in parsedDevDMs:
				print("* ERROR: we dont have",str(dm),"in parsed")
				exit(1)
		# sanity check part 2: ensures we have predictions for all of our parsed DMs
		for dm in parsedDevDMs:
			if dm not in predDevDMs:
				ref = self.corpus.dmToREF[dm]
				(d1,m1) = dm
				print("* ERROR: missing",str(dm),"and the doc has # mentions:",str(len(self.corpus.docToDMs[d1])))
				exit(1)
		print("# parsed:",str(len(parsedDevDMs)))
		print("# pred:",str(len(predDevDMs)))
		self.trainX, self.trainY = self.loadDynamicData(dirHalfToDMs, dirHalfToDMPredictions)

	def cluster(self, stoppingPoint):

		print("* in Cluster()")
		# loads test predictions
		predTestDMs = set()

		# stores predictions
		# NOTE: although i'm using the variable 'HM', this is robust; it can handle ECB data too,
		#       not just HDDCRP data (i had to name the variable something and couldn't name it DM and HM)
		dirHalfToHMPredictions = defaultdict(lambda : defaultdict(float))
		dirHalfToHMs = defaultdict(list) # used for ensuring our predictions included ALL valid HMs

		if self.testingPairs == None and self.testingPreds == None: # read the file
			print("* why are we in here?")
			exit(1)
			# sanity check:
			if self.args.useECBTest:
				print("* ERROR: we want to use ECBTest data, but we aren't passing it to FFNNWD")
				exit(1)

			# we know we are reading from HDDCRP predicted test mentions
			f = open(self.args.dataDir + "test_" + str(self.args.devDir) + ".txt")
			for line in f:
				hm1,hm2,pred = line.rstrip().split(",")
				hm1 = int(hm1)
				hm2 = int(hm2)
				pred = float(pred)
				doc_id = self.hddcrp_parsed.hm_idToHMention[hm1].doc_id
				doc_id2 = self.hddcrp_parsed.hm_idToHMention[hm2].doc_id
				if doc_id != doc_id2:
					print("ERROR: pairs are from diff docs")
					exit(1)
				if hm1 not in docToHMs[doc_id]:
					docToHMs[doc_id].append(hm1)
				if hm2 not in docToHMs[doc_id]:
					docToHMs[doc_id].append(hm2)
				docToHMPredictions[doc_id][(hm1,hm2)] = pred
				predTestDMs.add(hm1)
				predTestDMs.add(hm2)
			f.close()
		else: # use the passed-in Mentions (which could be ECB or HDDCRP format)
			for _ in range(len(self.testingPairs)):
				(dm1,dm2) = self.testingPairs[_]
				pred = self.testingPreds[_][0]
				print("testingPair:",str(dm1),"and",str(dm2))
				if self.args.useECBTest:
					doc_id1 = dm1[0]
					doc_id2 = dm2[0]
				else:
					doc_id1 = self.hddcrp_parsed.hm_idToHMention[dm1].doc_id
					doc_id2 = self.hddcrp_parsed.hm_idToHMention[dm2].doc_id

				extension1 = doc_id1[doc_id1.find("ecb"):]
				extension2 = doc_id2[doc_id2.find("ecb"):]
				dir_num1 = int(doc_id1.split("_")[0])
				dir_num2 = int(doc_id2.split("_")[0])				
				dirHalf1 = str(dir_num1) + extension1
				dirHalf2 = str(dir_num2) + extension2

				if dirHalf1 != dirHalf2:
					print("* ERROR, somehow, training pairs came from diff dir-halves")
					exit(1)

				if dm1 not in dirHalfToHMs[dirHalf1]:
					dirHalfToHMs[dirHalf1].append(dm1)
				if dm2 not in dirHalfToHMs[dirHalf1]: # not an error; dirHalf1 == dirHalf2
					dirHalfToHMs[dirHalf1].append(dm2)
				dirHalfToHMPredictions[dirHalf1][(dm1,dm2)] = pred
				predTestDMs.add(dm1)
				predTestDMs.add(dm2)

		# sanity check: ensures we are working w/ all of the DMs
		parsedDMs = set()
		if self.args.useECBTest:
			for d in self.helper.testingDirs:
				for doc in self.corpus.dirToDocs[d]:
					for dm in self.corpus.docToDMs[doc]:
						parsedDMs.add(dm)
						if dm not in predTestDMs:
							print("* ERROR: did not have the dm:",str(dm),"which was in our parsed ECB Test")
							exit(1)

			print("# predTestDMs:",str(len(predTestDMs)))
			print("# parsedDMs:",str(len(parsedDMs)))

			# sanity check part 2: ensures we have predictions for all of our parsed DMs
			for dm in parsedDMs:
				if dm not in predTestDMs:
					print("* ERROR: missing",str(dm),"from the predicted test set of DMs")
					exit(1)

		else: # hddcrp
			for hm in self.hddcrp_parsed.hm_idToHMention:
				if hm not in predTestDMs:
					print("* ERROR: predTestDMs is missing",str(hm))

					exit(1)
		print("# dms in test:",str(len(predTestDMs)))

		# now, the magic actually happens: time to cluster!
		ourClusterID = 0
		ourClusterSuperSet = {}

		goldenClusterID = 0
		goldenSuperSet = {}
		if self.args.useECBTest: # construct golden clusters
			for dirHalf in dirHalfToHMPredictions.keys():
				print("dirHalf:",dirHalf,"has",str(len(dirHalfToHMs[dirHalf])),"DMs")
				# ensures we have all DMs
				if len(dirHalfToHMs[dirHalf]) != len(self.corpus.dirHalfToHMs[dirHalf]):
					print("mismatch in DMs!! local dirHalfToHMs:",str(len(dirHalfToHMs[dirHalf])),"parsedCorpus:",str(len(self.corpus.dirHalfToHMs[dirHalf])))
					exit(1)

				# construct the golden truth for the current dirHalf
				for curREF in self.corpus.dirHalfREFToDMs[dirHalf].keys():
					tmp = set()
					for dm in self.corpus.dirHalfREFToDMs[dirHalf][curREF]:
						tmp.add(dm)
					goldenSuperSet[goldenClusterID] = tmp
					goldenClusterID += 1
		print("goldenSuperSet:",str(goldenSuperSet))
		for p in goldenSuperSet.keys():
			print("g:",str(g))
			for i in goldenSuperSet[g]:
				print("i:",i)
		for dirHalf in dirHalfToHMPredictions.keys():
			print("dirHalf:",str(dirHalf))
			numDMsInDirHalf = len(dirHalfToHMs[dirHalf])
			print("numDMsInDirHalf:",numDMsInDirHalf)
			print("# pairs in dirHalfToHMPredictions:",str(len(dirHalfToHMPredictions[dirHalf])))
			# stores all preds for the current dirHalf
			dirHalfPreds = []
			for pair in dirHalfToHMPredictions[dirHalf]:
				dirHalfPreds.append(dirHalfToHMPredictions[dirHalf][pair])

			# constructs our base clusters (singletons)
			ourDirHalfClusters = {} 
			for i in range(len(dirHalfToHMs[dirHalf])):
				hm = dirHalfToHMs[dirHalf][i]
				a = set()
				a.add(hm)
				ourDirHalfClusters[i] = a
				print("setting ourDirHalfClusters[",i,"] = ",str(a))
			# the following keeps merging until our shortest distance > stopping threshold,
			# or we have 1 cluster, whichever happens first

			# stores the cluster distances so that we don't have to do the expensive
			# computation every time
			clusterDistances = defaultdict(lambda : defaultdict(float))
			while len(ourDirHalfClusters.keys()) > 1:

				print("# clusters:",str(len(ourDirHalfClusters.keys())))
				# find best merge
				closestDist = 99999
				closestClusterKeys = (-1,-1)
				i = 0
				for c1 in ourDirHalfClusters.keys():
					for dm1 in ourDirHalfClusters[c1]:
						j = 0
						for c2 in ourDirHalfClusters.keys():
							if j > i:
								
								dist = -1

								# if we don't have it stored, it means it concerns the last merged/formed cluster
								# so let's compute it now and save it
								if clusterDistances[c1][c2] == None:
									if clusterDistances[c2][c1] != None: # ensure we didn't accidentally leave this one
										print("ERROR: missing clusterDistances")
										exit(1)
									X = []
									print("comparing dm1:",str(dm1),"to",str(ourDirHalfClusters[c2]))
									featureVec = self.getClusterFeatures(dm1, ourDirHalfClusters[c2], dirHalfToHMPredictions[dirHalf], numDMsInDirHalf)
									X.append(np.asarray(featureVec))
									X = np.asarray(X)
									# the first [0] is required to get into the surrounding array
									# the second [0] is to access the probability of not-match
									# so the lower it is means the higher the prob. of 'is-match' [1]
									dist = float(self.model.predict(X)[0][0])
									clusterDistances[c1][c2] = dist
									clusterDistances[c2][c1] = dist
								else: # we already have the distance, so let's just use it
									dist = clusterDistances[c1][c2]

								print(str(ourDirHalfClusters[c1]),str(ourDirHalfClusters[c2]),"dist:",str(dist))
								if dist < closestDist:
									closestDist = dist
									closestClusterKeys = (c1,c2)
									print("setting closestDist:",closestDist,"and closestClusterKeys:",str(closestClusterKeys))
							j += 1
					i += 1
				if closestDist > stoppingPoint:
					break
				newCluster = set()
				(c1,c2) = closestClusterKeys
				print("* merging:",ourDirHalfClusters[c1],"and",ourDirHalfClusters[c2],"dist:",closestDist)
				for _ in ourDirHalfClusters[c1]:
					newCluster.add(_)
				for _ in ourDirHalfClusters[c2]:
					newCluster.add(_)
				ourDirHalfClusters.pop(c1, None)
				ourDirHalfClusters.pop(c2, None)
				ourDirHalfClusters[c1] = newCluster
				
				# removes c1 and c2 from clusterDistances
				# so, on the next round, we only need to compute these new ones
				for i in range(len(ourDirHalfClusters.keys())):
					for j in range(len(ourDirHalfClusters.keys())):
						if j > i:
							clusterDistances[i][j] = None
							clusterDistances[j][i] = None
				sys.stdout.flush()
			# end of current dirHalf
			
			print("our final clustering of dirhalf:",str(dirHalf),"yielded # clusters:",str(len(ourDirHalfClusters.keys())))

			# goes through each cluster for the current dirHalf
			for i in ourDirHalfClusters.keys():

				# if True, remove REFs which aren't in the gold set, then remove singletons
				if self.ChoubeyFilter and not self.args.useECBTest:
					newCluster = set()
					for dm in ourDirHalfClusters[i]:
						muid = self.hddcrp_parsed.hm_idToHMention[dm].UID
						if muid in self.hddcrp_parsed.gold_MUIDToHMentions:
							newCluster.add(dm)
					if len(newCluster) > 0:
						ourClusterSuperSet[ourClusterID] = newCluster
						ourClusterID += 1
				else:
					ourClusterSuperSet[ourClusterID] = ourDirHalfClusters[i]
					ourClusterID += 1
		# end of going through every dirHalf
		#print("# our clusters:",str(len(ourClusterSuperSet)))
		return (ourClusterSuperSet, goldenSuperSet)

	def getAccuracy(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == np.argmax(preds, axis=1))

	def getAccuracy2(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == preds)

	def forwardprop(self, X, w_1, w_2, b_1, b_2):
		h1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(X, 1.0), w_1), b_1))
		return tf.add(tf.matmul(h1, w_2), b_2)

	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.1))

	def loadDynamicData(self, dirHalfToDMs, dirHalfToDMPredictions):
		
		# constructs a mapping of DIRHALF -> {REF -> DM}
		dirHalfREFToDMs = defaultdict(lambda : defaultdict(set))
		for dirHalf in dirHalfToDMs:
			for dm in dirHalfToDMs[dirHalf]:
				ref_id = self.corpus.dmToREF[dm]
				dirHalfREFToDMs[dirHalf][ref_id].add(dm)

		positiveData = []
		negativeData = []
		X = []
		Y = []

		# iterates through all dirHalves
		for dirHalf in dirHalfToDMs:

			numDMsInDirHalf = len(dirHalfToDMs[dirHalf])
			if numDMsInDirHalf == 1:
				print("* DIRHALF:",str(dirHalf),"HAS SINGLETON:",str(numDMsInDirHalf))
				continue

			# sanity check: ensures we have all predictions for the current dirHalf
			for dm1 in dirHalfToDMs[dirHalf]:
				for dm2 in dirHalfToDMs[dirHalf]:
					if dm1 == dm2:
						continue

					if (dm1,dm2) not in dirHalfToDMPredictions[dirHalf] and (dm2,dm1) not in dirHalfToDMPredictions[dirHalf]:
						print("* ERROR: we dont have dm1-dm2")
						exit(1)
			
			# pick a DM
			for dm1 in dirHalfToDMs[dirHalf]:
				gold_ref_id = self.corpus.dmToREF[dm1]
				# we can only pick a positive if there are other items in the cluster
				if len(dirHalfREFToDMs[dirHalf][gold_ref_id]) > 1:
					featureVec = self.getClusterFeatures(dm1, dirHalfREFToDMs[dirHalf][gold_ref_id], dirHalfToDMPredictions[dirHalf], numDMsInDirHalf)
					positiveData.append(featureVec)
					X.append(featureVec)
					Y.append([0,1])
				# looks for other clusters to compare DM1 to
				for other_ref_id in dirHalfREFToDMs[dirHalf].keys():
					if other_ref_id == gold_ref_id:
						continue
					if len(negativeData) < self.args.numNegPerPos * len(positiveData):
						featureVec = self.getClusterFeatures(dm1, dirHalfREFToDMs[dirHalf][other_ref_id], dirHalfToDMPredictions[dirHalf], numDMsInDirHalf)
						negativeData.append(featureVec)
						X.append(featureVec)
						Y.append([1,0])
		return (X,Y)

	# gets the features we care about -- how a DM relates to the passed-in cluster (set of DMs)
	def getClusterFeatures(self, dm1, allDMsInCandidateCluster, dirHalfToPredictions, numDMsInDirHalf):

		predsIn = []
		minPredIn = 9999
		maxPredIn = -1
		for dm2 in allDMsInCandidateCluster:

			if dm1 == dm2:
				continue
			if (dm1,dm2) in dirHalfToPredictions:
				pred = dirHalfToPredictions[(dm1,dm2)]
			elif (dm2,dm1) in dirHalfToPredictions:
				pred = dirHalfToPredictions[(dm2,dm1)]
			else:
				print("* ERROR: prediction doesn't exist")
				exit(1)

			if pred < minPredIn:
				minPredIn = pred
			if pred > maxPredIn:
				maxPredIn = pred
			predsIn.append(pred)

		clusterSizePercentage = float(len(allDMsInCandidateCluster)) / float(numDMsInDirHalf)
		avgPredIn = sum(predsIn) / len(predsIn)
		featureVec = [minPredIn, avgPredIn, maxPredIn, clusterSizePercentage] # A
		return featureVec