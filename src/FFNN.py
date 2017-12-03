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
class FFNN:
	def __init__(self, args, corpus, helper, hddcrp_parsed, dev_pairs=None, dev_preds=None):

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
		self.testingPairs = ""
		self.testX = ""
		self.testY = ""

		self.model = None # filled in via train()

		self.createTraining(dev_pairs, dev_preds) # loads training/test data

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
		#evaluation = self.model.evaluate(self.testX, self.testY, verbose=1)
		#return (self.testingPairs, preds, self.testY)

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
		print("* in createTraining")

		# sanity check part 1: ensure all dev DMs are accounted for (that we have prediction values for all)
		parsedDevDMs = set()
		for d in self.helper.devDirs:
			for doc in self.corpus.dirToDocs[d]:
				for dm in self.corpus.docToDMs[doc]:
					parsedDevDMs.add(dm)

		# loads dev predictions via CCNN's pairwise predictions, or a saved file
		devPredictions = {}
		predDevDMs = set()
		if dev_pairs == None and dev_preds == None: # read the file
			f = open(self.args.dataDir + "dev_" + str(self.args.devDir) + ".txt")
			for line in f:
				d1,m1,d2,m2,pred = line.rstrip().split(",")
				m1 = int(m1)
				m2 = int(m2)
				dm1 = (d1,m1)
				dm2 = (d2,m2)
				pred = float(pred)
				devPredictions[(dm1,dm2)] = pred
				predDevDMs.add(dm1)
				predDevDMs.add(dm2)
			f.close()		
		else: # read the CCNN predictions
			for _ in range(len(dev_pairs)):
				(dm1,dm2) = dev_pairs[_]
				predDevDMs.add(dm1)
				predDevDMs.add(dm2)
				devPredictions[dev_pairs[_]] = dev_preds[_][0]

		# sanity check part 2: ensure all dev DMs are accounted for (that we have prediction values for all)
		# AND it maps DMs to a per-doc access
		docToPredDevDMs = defaultdict(set)
		for dm in predDevDMs:
			if dm not in parsedDevDMs:
				print("we dont have",str(dm),"in parsed")
			else: # we have the DM parsed, so we can look up its doc
				(doc_id,m_id) = dm
				docToPredDevDMs[doc_id].add(dm)

		'''
		# loads test predictions
		testPredictions = {}
		predTestDMs = set()
		f = open(self.args.dataDir + "test_" + str(self.args.devDir) + ".txt")
		for line in f:
			hm1,hm2,pred = line.rstrip().split(",")
			hm1 = int(hm1)
			hm2 = int(hm2)
			pred = float(pred)
			testPredictions[(hm1,hm2)] = pred
			predTestDMs.add(hm1)
			predTestDMs.add(hm2)
		f.close()
		# makes testing Doc->DMs
		docToPredTestDMs = defaultdict(set)
		for hm in predTestDMs:
			doc_id = self.hddcrp_parsed.hm_idToHMention[hm].doc_id
			docToPredTestDMs[doc_id].add(hm)
		'''
		for dm in parsedDevDMs:
			if dm not in predDevDMs:
				ref = self.corpus.dmToREF[dm]
				(d1,m1) = dm
				print("missing",str(dm),"and the doc has # mentions:",str(len(self.corpus.docToDMs[d1])))
		print("# parsed:",str(len(parsedDevDMs)))
		print("# pred:",str(len(predDevDMs)))

		# the following is for the static-clustering; that is, we make predictions before
		# any clustering is done.  we don't make FFNN clustering decisions.
		# instead, we just make weights which will use agglomerative to decide
		#_, self.trainX, self.trainY = self.loadStaticData(docToPredDevDMs, devPredictions, False)
		#self.testingPairs, self.testX, self.testY = self.loadStaticData(docToPredTestDMs, testPredictions, True)

		self.trainX, self.trainY = self.loadDynamicData(docToPredDevDMs, devPredictions)
		#self.loadDynamicData(docToPredTestDMs, testPredictions, True)

	def getAccuracy(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == np.argmax(preds, axis=1))

	def getAccuracy2(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == preds)

	def forwardprop(self, X, w_1, w_2, b_1, b_2):
		h1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(X, 1.0), w_1), b_1))
		return tf.add(tf.matmul(h1, w_2), b_2)

	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.1))

	def loadDynamicData(self, docToPredDMs, predictions):
		# constructs a mapping of DOC -> {REF -> DM}
		docToREFDMs = defaultdict(lambda : defaultdict(set))
		docToDMs = defaultdict(set)
		for doc_id in docToPredDMs:
			for dm in docToPredDMs[doc_id]:
				ref_id = self.corpus.dmToREF[dm]
				docToREFDMs[doc_id][ref_id].add(dm)
				docToDMs[doc_id].add(dm)

		positiveData = []
		negativeData = []
		X = []
		Y = []
		for doc_id in docToREFDMs:
			if len(docToDMs[doc_id]) == 1:
				print("* DOC:",str(doc_id),"HAS SINGLETON:",str(docToDMs[doc_id]))
				continue


			# pre-processes: makes a sorted list of all predictions (non-dupes) in the doc
			allDocPreds = []
			added = set()
			for dm1 in docToPredDMs[doc_id]:
				for dm2 in docToPredDMs[doc_id]:
					if dm1 == dm2:
						continue
					pair = None
					if (dm1,dm2) in predictions:
						pair = (dm1,dm2)
					elif (dm2,dm1) in predictions:
						pair = (dm2,dm1)
					else:
						print(len(added))
						print("* ERROR: we dont have dm1-dm2")
						exit(1)
					if pair not in added:
						allDocPreds.append(predictions[pair])
						added.add(pair)
			sorted_preds = sorted(allDocPreds)
			
			# pick a DM
			for dm1 in docToDMs[doc_id]:
				gold_ref_id = self.corpus.dmToREF[dm1]
				# we can only pick a positive if there are other items in the cluster
				if len(docToREFDMs[doc_id][gold_ref_id]) > 1:
					featureVec = self.getClusterFeatures(dm1, docToREFDMs[doc_id][gold_ref_id], sorted_preds, predictions)
					positiveData.append(featureVec)
					X.append(featureVec)
					Y.append([0,1])
				# looks for other clusters to compare DM1 to
				for other_ref_id in docToREFDMs[doc_id].keys():
					if other_ref_id == gold_ref_id:
						continue
					if len(negativeData) < self.args.numNegPerPos * len(positiveData):
						featureVec = self.getClusterFeatures(dm1, docToREFDMs[doc_id][other_ref_id], sorted_preds, predictions)
						negativeData.append(featureVec)
						X.append(featureVec)
						Y.append([1,0])
		return (X,Y)
	# gets the features we care about -- how a DM relates to the passed-in cluster
	def getClusterFeatures(self, dm1, allDMsInCluster, sorted_preds, predictions):
		minPred = 999
		preds = []
		for dm2 in allDMsInCluster:
			if dm1 == dm2:
				continue
			if (dm1,dm2) in predictions:
				pred = predictions[(dm1,dm2)]
			elif (dm2,dm1) in predictions:
				pred = predictions[(dm2,dm1)]
			else:
				print("* ERROR: prediction doesn't exist")
				exit(1)
			if pred < minPred:
				minPred = pred
			preds.append(pred)

		avgPred = sum(preds) / len(preds)
		numItems = len(preds)
		indexAboveMin = 0
		indexAboveAvg = 0
		for _ in range(len(sorted_preds)):
			if sorted_preds[_] < minPred:
				indexAboveMin += 1
			if sorted_preds[_] < avgPred:
				indexAboveAvg += 1
		percentageBelowMin = float(indexAboveMin) / len(sorted_preds)
		percentageBelowAvg = float(indexAboveAvg) / len(sorted_preds)
		featureVec = [minPred, avgPred] # A 
		#featureVec = [minPred, avgPred, numItems] # B
		#featureVec = [percentageBelowMin, percentageBelowAvg] # C
		#featureVec = [percentageBelowMin, percentageBelowAvg, numItems] # D
		#featureVec = [minPred, avgPred, percentageBelowMin, percentageBelowAvg] # E (or include numItems if it ever helps)
		return featureVec

	def loadStaticData(self, docToPredDMs, predictions, isHDDCRP):
		addedPairs = set()
		pairs = []
		X = []
		Y = []
		numP = 0
		numN = 0

		# pre-processes; constructs the distribution of all links from a given m
		dmToLinkDistribution = defaultdict(lambda: [0] * 10)
		docToListDistribution = defaultdict(lambda: [0] * 10)
		for p in predictions:
			(dm1,dm2) = p
			pred = predictions[p]
			binNum = math.floor(min(0.9999,pred)*10)

			if isHDDCRP: # meaning testset
				doc_id = self.hddcrp_parsed.hm_idToHMention[dm1].doc_id
			else:
				(doc_id,m_id) = dm1

			dmToLinkDistribution[dm1][binNum] += 1
			dmToLinkDistribution[dm2][binNum] += 1
			docToListDistribution[doc_id][binNum] += 1

		# normalizes mention weights into a distribution
		for m in dmToLinkDistribution:
			mSum = sum(dmToLinkDistribution[m])
			li = []
			for _ in dmToLinkDistribution[m]:
				li.append(float(_/mSum))
			dmToLinkDistribution[m] = li

		# normalizes doc weights into a distribution
		for doc_id in docToListDistribution:
			docSum = sum(docToListDistribution[doc_id])
			li = []
			for _ in docToListDistribution[doc_id]:
				li.append(float(_/docSum))
			docToListDistribution[doc_id] = li

		for doc in docToPredDMs:
			for dm1 in docToPredDMs[doc]:
				positivePairs = set()
				negativePairs = set()
				for dm2 in docToPredDMs[doc]:

					# if we have a singleton, let's use it
					if len(docToPredDMs[doc]) > 1 and dm1 == dm2:
						continue
					# we haven't yet added it to then positive or negative examples
					# (don't want duplicate training pairs)
					if (dm1,dm2) not in addedPairs and (dm2,dm1) not in addedPairs:				
						curX = []
						addExample = True # will only become False if the # of neg devs is greater than our param
						if (dm1,dm2) in predictions:
							curX.append(predictions[(dm1,dm2)])
						elif (dm2,dm1) in predictions:
							curX.append(predictions[(dm2,dm1)])
						else:
							print("* ERROR: dm1,dm2 or reverse aren't in predictions")
							exit(1)

						curX += docToListDistribution[doc]
						curX += dmToLinkDistribution[dm1]

						curY = []
						if isHDDCRP:
							ref1 = self.hddcrp_parsed.hm_idToHMention[dm1].ref_id
							ref2 = self.hddcrp_parsed.hm_idToHMention[dm2].ref_id
							if ref1 == ref2:
								curY = [0,1]
								numP += 1
							else:
								curY = [1,0]
								numN += 1

						else: # ECB Corpus Mentions
							# positive
							if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]: 
								curY = [0,1]
								numP += 1
							else:
								if numN >= self.args.numNegPerPos * numP:
									addExample = False
								else:
									curY = [1,0]
									numN += 1

						if addExample:
							# adds training
							X.append(curX)
							Y.append(curY)

							# just ensures we don't add the same training (pos or neg) twice
							addedPairs.add((dm1,dm2))
							addedPairs.add((dm2,dm1))

							pairs.append((dm1,dm2))
		print("numP:",str(numP))
		print("numN:",str(numN))
		return (pairs, X, Y)
