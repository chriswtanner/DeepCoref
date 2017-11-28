import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
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
	def __init__(self, args, corpus, helper, hddcrp_parsed):

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

		self.createTraining() # loads training/test data

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

	def run(self):
		model = Sequential()
		model.add(Dense(units=self.hidden_size, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		model.add(Activation('sigmoid'))
		model.add(Dense(units=self.outputDim, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		model.add(Activation('softmax'))

		if self.FFNNOpt == "rms":
			opt = RMSprop()
		elif self.FFNNOpt == "adam":
			opt = Adam(lr=0.001)
		elif self.FFNNOpt == "adagrad":
			opt = Adagrad()
		else:
			print("* ERROR: invalid CCNN optimizer")
			exit(1)
		model.compile(loss=self.weighted_binary_crossentropy,optimizer=opt,metrics=['accuracy'])
		model.fit(self.trainX, self.trainY, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
		evaluation = model.evaluate(self.testX, self.testY, verbose=1)
		preds = model.predict(self.testX, verbose=1)
		print("\nevaluation:",str(evaluation))
		print("test acc:",str(self.getAccuracy(preds, self.testY)))
		print("f1:",str(self.calculateF1(preds)))
		return (self.testingPairs, preds, self.testY)

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

	def createTraining(self):
		print("* in createTraining")

		# sanity check part 1: ensure all dev DMs are accounted for (that we have prediction values for all)
		parsedDevDMs = set()
		for d in self.helper.devDirs:
			for doc in self.corpus.dirToDocs[d]:
				for dm in self.corpus.docToDMs[doc]:
					parsedDevDMs.add(dm)

		# loads dev predictions
		devPredictions = {}
		predDevDMs = set()
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

		# sanity check part 2: ensure all dev DMs are accounted for (that we have prediction values for all)
		docToPredDevDMs = defaultdict(set)
		for dm in predDevDMs:
			if dm not in parsedDevDMs:
				print("we dont have",str(dm),"in parsed")
			else: # we have the DM parsed, so we can look up its doc
				(d1,m1) = dm
				docToPredDevDMs[d1].add(dm)

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

		for dm in parsedDevDMs:
			if dm not in predDevDMs:
				ref = self.corpus.dmToREF[dm]
				(d1,m1) = dm
				print("missing",str(dm),"and the doc has # mentions:",str(len(self.corpus.docToDMs[d1])))
		print("# parsed:",str(len(parsedDevDMs)))
		print("# pred:",str(len(predDevDMs)))

		_, self.trainX, self.trainY = self.loadData(docToPredDevDMs, devPredictions, False)
		self.testingPairs, self.testX, self.testY = self.loadData(docToPredTestDMs, testPredictions, True)

	def getAccuracy(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == np.argmax(preds, axis=1))

	def getAccuracy2(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == preds)

	def forwardprop(self, X, w_1, w_2, b_1, b_2):
		h1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(X, 1.0), w_1), b_1))
		return tf.add(tf.matmul(h1, w_2), b_2)

	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.1))

	def loadData(self, docToPredDMs, predictions, isHDDCRP):
		addedPairs = set()
		pairs = []
		X = []
		Y = []
		numP = 0
		numN = 0
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
