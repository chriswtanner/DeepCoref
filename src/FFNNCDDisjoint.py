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
import time
from itertools import product
from sortedcontainers import SortedDict
class FFNNCDDisjoint: # this class handles CCNN CD model, but training/testing is done on a Cluster basis, disjoint only
	def __init__(self, args, corpus, helper, hddcrp_parsed, dev_pairs=None, dev_preds=None, testing_pairs=None, testing_preds=None):

		self.ChoubeyFilter = False # if True, remove the False Positives.  Missed Mentions still exist though.
		self.numCorpusSamples = 1

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
		print("in FFNNCDDisjoint's createTraining()")

		parsedDevDMs = set()
		for d in self.helper.devDirs:
			for doc in self.corpus.dirToDocs[d]:
				for dm in self.corpus.docToDMs[doc]:
					parsedDevDMs.add(dm)

		# loads dev predictions via CCNN's pairwise predictions, or a saved file
		dirHalfToDMPredictions = defaultdict(lambda : defaultdict(float))
		dirHalfToDMs = defaultdict(set) # used for ensuring our predictions included ALL valid DMs
		predDevDMs = set() # only used for sanity check
	
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

			dirHalfToDMs[dirHalf1].add(dm1)
			dirHalfToDMs[dirHalf2].add(dm2) # correct; dirHalf1 == dirHalf2

			dirHalfToDMPredictions[dirHalf1][(dm1,dm2)] = pred
			predDevDMs.add(dm1)
			predDevDMs.add(dm2)

		# sanity check part 1: ensures we have parsed all of our predicted DMs
		docToPredDevDMs = defaultdict(set)
		print("# parsed:",str(len(parsedDevDMs)))
		print("# pred:",str(len(predDevDMs)))
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

		self.trainX, self.trainY = self.loadDynamicData(dirHalfToDMs, dirHalfToDMPredictions)

	def clusterWDClusters(self, wdClusters, stoppingPoint2):
		print("* FFNNCDDisjoint - clusterWDs()")
		start_time = time.time()
		# loads test predictions
		predTestDMs = set()

		# stores predictions
		# NOTE: although i'm using the variable 'HM', this is robust; it can handle ECB data too,
		#       not just HDDCRP data (i had to name the variable something and couldn't name it DM and HM)
		dirHalfToHMPredictions = defaultdict(lambda : defaultdict(float))
		dirHalfToHMs = defaultdict(list) # used for ensuring our predictions included ALL valid HMs

		# (STEP 2)
		# maps the wdClusters to their dirHalf's -- so we know what our candidate clusters are
		dirHalfToWDClusterNums = defaultdict(set)
		for clusterNum in wdClusters.keys():
			cluster = wdClusters[clusterNum]
			keys = set()
			oneKey = None
			oneDoc = None
			print("wdCluster:",str(clusterNum))
			for hm in cluster:
				if self.args.useECBTest:
					doc_id = hm[0]
				else:
					doc_id = self.hddcrp_parsed.hm_idToHMention[hm].doc_id

				if doc_id != oneDoc and oneDoc != None:
					print("* ERROR: multiple docs within the wd base cluster")
					exit(1)    
				oneDoc = doc_id
				extension = doc_id[doc_id.find("ecb"):]
				dir_num = int(doc_id.split("_")[0])
				key = str(dir_num) + extension
				keys.add(key)
				oneKey = key

				print("key:",key,"clusterNum:",clusterNum)
			if len(keys) != 1:
				print("* ERROR: cluster had # keys:",str(len(keys)))

			dirHalfToWDClusterNums[oneKey].add(clusterNum)

		for dirHalf in dirHalfToWDClusterNums:
			print("dirHalfToWDClusterNums[dirHalf]:",dirHalf,"=",dirHalfToWDClusterNums[dirHalf])

		print("per WD clusters, we have # dirHalfs:",str(len(dirHalfToWDClusterNums.keys())))
		print("per the corpus, we have # dirHalfs:",str(len(self.corpus.dirHalfREFToDMs.keys())))

		# (STEP 3) stores dirHalf predictions
		# use the passed-in Mentions (which could be ECB or HDDCRP format)
		for _ in range(len(self.testingPairs)):
			(dm1,dm2) = self.testingPairs[_]
			pred = self.testingPreds[_][0]
			#print("testingPair:",str(dm1),"and",str(dm2))
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

		print("per CCNN, #dirHalfs in dirHalfToHMs:",str(len(dirHalfToHMs.keys())))
		print("per corpus, #dirHalfs in corpus.dirHalfToHMs:",str(len(self.corpus.dirHalfToHMs.keys())))

		# (STEP 4): sanity check: ensures we have all of the DMs
		for dirHalf in dirHalfToHMs:
			print("dirHalf:",dirHalf,"dirHalfToHMs:",len(dirHalfToHMs[dirHalf]),"self.corpus.dirHalfToHMs:",len(self.corpus.dirHalfToHMs[dirHalf]))
			if self.args.useECBTest:
				if len(dirHalfToHMs[dirHalf]) != len(self.corpus.dirHalfToHMs[dirHalf]):
					print("* ERROR: differing # of DMs b/w CCNN and the Corpus")
					exit(1)

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

		else: # hddcrp
			for hm in self.hddcrp_parsed.hm_idToHMention:
				parsedDMs.add(hm)
				if hm not in predTestDMs:
					print("* ERROR: predTestDMs is missing",str(hm))
					exit(1)

		# sanity check part 2: ensures we have parsed for all of our predDMs
		for dm in predTestDMs:
			if dm not in parsedDMs:
				print("* ERROR: missing",str(dm),"from the parsed set of DMs")
				exit(1)

		print("# dms in test:",str(len(predTestDMs)))
		print("# dms in parsed:",str(len(parsedDMs)))
		# ---- END OF STEP 4

		# now, the magic actually happens: time to cluster!
		ourClusterID = 0
		ourClusterSuperSet = {}
		
		# (STEP 4B - OPTIONAL): # construct the golden truth for the current dir-half
		goldenClusterID = 0
		goldenSuperSet = {}
		if self.args.useECBTest: # construct golden clusters
			for dirHalf in dirHalfToHMPredictions.keys():

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

		# (STEP 5): makes base clusters and then does clustering!
		for dirHalf in dirHalfToWDClusterNums.keys():

			print("dirHalf:",str(dirHalf))
			numDMsInDirHalf = len(dirHalfToHMs[dirHalf])
			print("numDMsInDirHalf:",numDMsInDirHalf)
			print("# pairs in dirHalfToHMPredictions:",str(len(dirHalfToHMPredictions[dirHalf])))

			# constructs our base clusters (singletons)
			ourDirHalfClusters = {} 
			clusterNumToDocs = defaultdict(set)
			
			highestClusterNum = 0
			# sets base clusters to be the WD clusters
			for wdClusterNum in dirHalfToWDClusterNums[dirHalf]:
				a = set()
				for dm in wdClusters[wdClusterNum]:
					a.add(dm)
					if self.args.useECBTest:
						doc_id = dm[0]
					else:
						doc_id = self.hddcrp_parsed.hm_idToHMention[dm].doc_id
					clusterNumToDocs[highestClusterNum].add(doc_id)
				ourDirHalfClusters[highestClusterNum] = a
				highestClusterNum += 1
			print("# WD base clusters:",str(len(ourDirHalfClusters)))
			for base in ourDirHalfClusters:
				print("base:",base)
				for hm in ourDirHalfClusters[base]:
					print(str(self.hddcrp_parsed.hm_idToHMention[hm]))


			# stores the cluster distances so that we don't have to do the expensive
			# computation every time
			# seed the distances
			clusterDistances = SortedDict()
			added = set()
			for c1 in ourDirHalfClusters.keys():
				docsInC1 = clusterNumToDocs[c1]
				print("c1:",str(c1),"with docs:",docsInC1)
				for hm in ourDirHalfClusters[c1]:
					print(self.hddcrp_parsed.hm_idToHMention[hm])

				for c2 in ourDirHalfClusters.keys():
					if (c1,c2) in added or (c2,c1) in added or c1 == c2:
						continue
					docsInC2 = clusterNumToDocs[c2]

					# only consider merging clusters that are disjoint in their docs
					containsOverlap = False
					for d1 in docsInC1:
						if d1 in docsInC2:
							containsOverlap = True
							break
					if containsOverlap:
						continue

					if len(docsInC1) != 1 or len(docsInC2) != 1:
						print("* ERROR, a basecluster has more than 1 doc",docsInC1,docsInC2)
						exit(1)

					c1Size = len(ourDirHalfClusters[c1])
					c2Size = len(ourDirHalfClusters[c2])
					potentialSizePercentage = float(c1Size + c2Size) / float(numDMsInDirHalf)
					X = []
					featureVec = self.getClusterFeatures(ourDirHalfClusters[c1], ourDirHalfClusters[c2], dirHalfToHMPredictions[dirHalf], potentialSizePercentage)
					X.append(np.asarray(featureVec))
					X = np.asarray(X)
					dist = float(self.model.predict(X)[0][0])
					print("c2:",str(c2),"with docs:",docsInC2)
					for hm in ourDirHalfClusters[c2]:
						print(self.hddcrp_parsed.hm_idToHMention[hm])
					print("dist:",str(dist),"size:",potentialSizePercentage)
					#print("c1:",str(ourDirHalfClusters[c1]),"c2:",str(ourDirHalfClusters[c2]),"=",dist,potentialSizePercentage)
					if dist in clusterDistances:
						clusterDistances[dist].append((c1,c2))
					else:
						clusterDistances[dist] = [(c1,c2)]
					added.add((c1,c2))
			print("# in clusterDistances:",str(len(added)))

			'''
			NOTE currently, the only items distances we store are ones that could be merged
			so, i just pick the closest one, update the clusterNumToDocs, update the clusters,
			and add distances but only the ones of valid candidate clusters -- so, ones that dont 
			have the same doc contained (no overlap)
			ALSO, sometimes we could run out of valid clusters to merge, so handle this somewhere
			'''
			bad = set()
			cluster_start_time = time.time()
			while len(ourDirHalfClusters.keys()) > 1:
				searchForShortest = True
				shortestPair = None
				shortestDist = 99999
				while searchForShortest:
					(k,values) = clusterDistances.peekitem(0)
					newList = []
					for (c1,c2) in values:
						if c1 not in bad and c2 not in bad:
							newList.append((c1,c2))
					if len(newList) > 0: # not empty, yay, we don't have to keep searching
						searchForShortest = False
						shortestPair = newList.pop(0) # we may be making the list have 0 items now
						shortestDist = k
					if len(newList) > 0: # let's update the shortest distance's pairs
						clusterDistances[k] = newList
					else: # no good items, let's remove it from the dict
						del clusterDistances[k]

				if shortestDist > stoppingPoint2:
					break

				
				(c1,c2) = shortestPair
				bad.add(c1)
				bad.add(c2)

				print("shortestDist:",shortestDist,"merging",c1,"(",ourDirHalfClusters[c1],"),and",c2,"(",ourDirHalfClusters[c2],")")
				print("c1 to merge")
				for hm in ourDirHalfClusters[c1]:
					print(self.hddcrp_parsed.hm_idToHMention[hm])
				print("c2 to merge")
				for hm in ourDirHalfClusters[c2]:
					print(self.hddcrp_parsed.hm_idToHMention[hm])

				# create new cluster w/ its sub cluster's DMs
				newCluster = set()
				for _ in ourDirHalfClusters[c1]:
					newCluster.add(_)
				for _ in ourDirHalfClusters[c2]:
					newCluster.add(_)
				
				# remove the clusters
				ourDirHalfClusters.pop(c1,None)
				ourDirHalfClusters.pop(c2,None)

				# adds new cluster and its storage of docs
				ourDirHalfClusters[highestClusterNum] = newCluster
				newDocSet = set()
				for _ in clusterNumToDocs[c1]:
					newDocSet.add(_)
				for _ in clusterNumToDocs[c2]:
					newDocSet.add(_)
				clusterNumToDocs.pop(c1, None)
				clusterNumToDocs.pop(c2, None)
				clusterNumToDocs[highestClusterNum] = newDocSet
				

				newClusterSize = len(newCluster)

				print("new cluster [",highestClusterNum"]:")
				for hm in newCluster:
					print(self.hddcrp_parsed.hm_idToHMention[hm])
				# compute new distance values between this new cluster and all other valid (disjoint) clusters
				for c1 in ourDirHalfClusters:
					
					# don't consider the newCluster to itself (which has been stored in ourDirHalfClusters)
					if c1 == newCluster:
						continue
					docsInC1 = clusterNumToDocs[c1]

					containsOverlap = false
					for d in docsInC1:
						if d in newDocSet:
							containsOverlap = true
							break

					# skip to the next candidate cluster
					if containsOverlap:
						continue

					c1Size = len(ourDirHalfClusters[c1])
					c2Size = newClusterSize
					potentialSizePercentage = float(c1Size + newClusterSize) / float(numDMsInDirHalf)
					
					featureVec = self.getClusterFeatures(ourDirHalfClusters[c1], newCluster, dirHalfToHMPredictions[dirHalf], potentialSizePercentage)
					X = []
					X.append(np.asarray(featureVec))
					X = np.asarray(X)
					dist = float(self.model.predict(X)[0][0])
					if dist in clusterDistances:
						clusterDistances[dist].append((c1,highestClusterNum))
					else:
						clusterDistances[dist] = [(c1,highestClusterNum)]
				highestClusterNum += 1
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
			print("cur dirhalf took:",str((time.time() - cluster_start_time)),"seconds")
		# end of going through every dirHalf
		#print("# our clusters:",str(len(ourClusterSuperSet)))
		print("*** CLUSTERING TOOK",str((time.time() - start_time)),"SECONDS")
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

	# NOTE: DMs could be HMs; i handle for both
	def loadDynamicData(self, dirHalfToDMs, dirHalfToDMPredictions):
		
		# constructs a mapping of DIRHALF -> {REF -> DM}
		dirHalfREFToDMs = defaultdict(lambda : defaultdict(set))
		dirHalfREFToDocs = defaultdict(lambda : defaultdict(set))
		docREFToDMs = defaultdict(lambda : defaultdict(set))
		dirHalfDocToREFs = defaultdict(lambda : defaultdict(set))

		for dirHalf in dirHalfToDMs:
			for dm in dirHalfToDMs[dirHalf]:
				ref_id = self.corpus.dmToREF[dm]
				
				doc_id = dm[0]
				dirHalfREFToDMs[dirHalf][ref_id].add(dm)
				dirHalfREFToDocs[dirHalf][ref_id].add(doc_id)
				docREFToDMs[doc_id][ref_id].add(dm)
				dirHalfDocToREFs[dirHalf][doc_id].add(ref_id)

		positiveDataCount = 0
		negativeDataCount = 0
		X = []
		Y = []

		for i in range(self.numCorpusSamples):

			# iterates through all dirHalves
			for dirHalf in dirHalfToDMs:
				numDMsInDirHalf = len(dirHalfToDMs[dirHalf])
				if numDMsInDirHalf == 1:
					print("* DIRHALF:",str(dirHalf),"HAS SINGLETON:",str(numDMsInDirHalf))
					exit(1)

				# sanity check: ensures we have all predictions for the current dirHalf
				for dm1 in dirHalfToDMs[dirHalf]:
					doc_id1 = dm1[0]
					for dm2 in dirHalfToDMs[dirHalf]:
						doc_id2 = dm2[0]
						if dm1 == dm2 or doc_id1 == doc_id2:
							continue

						if (dm1,dm2) not in dirHalfToDMPredictions[dirHalf] and (dm2,dm1) not in dirHalfToDMPredictions[dirHalf]:
							print("* ERROR: we dont have",str(dm1),str(dm2),"in dirHalfToDMPredictions")
							print("dirHalfToDMPredictions[dirHalf]:",str(dirHalfToDMPredictions[dirHalf]))
							exit(1)
				
				# looks through each doc
				for doc_id in dirHalfDocToREFs[dirHalf].keys():
					# looks through each REF
					for ref_id in docREFToDMs[doc_id].keys():
						# ensures other docs contain the ref
						numDocsContainingRef = len(dirHalfREFToDocs[dirHalf][ref_id])
						if numDocsContainingRef == 1:
							continue

						curCluster = set()
						for dm in docREFToDMs[doc_id][ref_id]:
							curCluster.add(dm)
						numDesiredDocsInPseudoGoldCluster = random.randint(1,numDocsContainingRef-1)
						docsInPseudoGoldCluster = set() 
						while len(docsInPseudoGoldCluster) < numDesiredDocsInPseudoGoldCluster:
							randDoc = random.sample(dirHalfREFToDocs[dirHalf][ref_id],1)[0]
							if randDoc != doc_id:
								docsInPseudoGoldCluster.add(randDoc)
						pseudoCluster = set()
						for otherDoc in docsInPseudoGoldCluster:
							for dm in docREFToDMs[otherDoc][ref_id]:
								pseudoCluster.add(dm)

						curClusterSize = len(curCluster)
						pseudoClusterSize = len(pseudoCluster)
						potentialSizePercentage = float(curClusterSize + pseudoClusterSize) / float(numDMsInDirHalf)
						featureVec = self.getClusterFeatures(curCluster, pseudoCluster, dirHalfToDMPredictions[dirHalf], potentialSizePercentage)
						positiveDataCount += 1
						X.append(featureVec)
						Y.append([0,1])

						# constructs negative sample clusters
						while negativeDataCount < self.args.numNegPerPos * positiveDataCount:
							other_ref = random.sample(dirHalfREFToDMs[dirHalf].keys(),1)[0]
							if other_ref == ref_id:
								continue
							numDocsContainingOtherRef = len(dirHalfREFToDocs[dirHalf][other_ref])
							if doc_id in dirHalfREFToDocs[dirHalf][other_ref]:
								numDocsContainingOtherRef = numDocsContainingOtherRef - 1
							if numDocsContainingOtherRef < 1:
								continue
							numDesiredDocsInPseudoBadCluster = random.randint(1,numDocsContainingOtherRef)
							docsInPseudoBadCluster = set()
							while len(docsInPseudoBadCluster) < numDesiredDocsInPseudoBadCluster:
								randDoc = random.sample(dirHalfREFToDocs[dirHalf][other_ref],1)[0]
								if randDoc != doc_id:
									docsInPseudoBadCluster.add(randDoc)
							pseudoBadCluster = set()
							for otherDoc in docsInPseudoBadCluster:
								for dm in docREFToDMs[otherDoc][other_ref]:
									pseudoBadCluster.add(dm)
							pseudoBadClusterSize = len(pseudoBadCluster)
							potentialSizePercentage = float(curClusterSize + pseudoBadClusterSize) / float(numDMsInDirHalf)
							featureVec = self.getClusterFeatures(curCluster, pseudoBadCluster, dirHalfToDMPredictions[dirHalf], potentialSizePercentage)
							negativeDataCount += 1
							X.append(featureVec)
							Y.append([1,0])
		print("positiveDataCount:",positiveDataCount)
		print("negativeDataCount:",negativeDataCount)
		return (X,Y)

	# gets the features we care about -- how a DM relates to the passed-in cluster (set of DMs)
	def getClusterFeatures(self, cluster1, cluster2, dmPredictions, clusterSizePercentage):
		dists = []

		for dm1 in cluster1:
			for dm2 in cluster2:
				if dm1 == dm2 or cluster1 == cluster2:
					continue
				if (dm1,dm2) in dmPredictions:
					pred = dmPredictions[(dm1,dm2)]
				elif (dm2,dm1) in dmPredictions:
					pred = dmPredictions[(dm2,dm1)]
				else:
					print("dmPredictions:",str(dmPredictions))
					print("* ERROR: prediction doesn't exist",str(dm1),str(dm2))
					exit(1)
				dists.append(pred)
		minDist = min(dists)
		avgDist = sum(dists) / len(dists)
		maxDist = max(dists)
		featureVec = [minDist, avgDist, maxDist, clusterSizePercentage] # A
		return featureVec