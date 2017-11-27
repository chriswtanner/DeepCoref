import numpy as np
import tensorflow as tf
import random
import keras
import os
import sys
from tensorflow.python.client import device_lib
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

		self.createTraining()

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
		for dm in predDevDMs:
			if dm not in parsedDevDMs:
				print("we dont have",str(dm),"in parsed")
		for dm in parsedDevDMs:
			if dm not in predDevDMs:
				ref = self.corpus.dmToREF[dm]
				(d1,m1) = dm
				print("missing",str(dm),"and the doc has # mentions:",str(len(self.corpus.docToDMs[d1])))
		print("# parsed:",str(len(parsedDevDMs)))
		print("# pred:",str(len(predDevDMs)))

		TODO: 
			- make a separate FFNN class, which works on fake data
			- compare it to keras model
			- incorporate it in this class; ensure it works
			- make our data that format (training + testing)
