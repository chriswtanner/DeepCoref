import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils

import tensorflow as tf
import numpy as np
import random
import sys
import os
import math
class FFNNTF:

	def __init__(self):
		# params
		self.x_size = 10
		self.hidden_size = 50
		self.y_size = 2
		self.lr = 0.005
		self.num_epochs = 25
		self.dataDim = 10
		self.trainingSize = 102
		self.testingSize = 50
		self.batch_size = 5
		self.trainX = []
		self.trainY = []
		self.testX = []
		self.testY = []
		# ---------

		self.createData()

		model = Sequential()
		model.add(Dense(units=self.hidden_size, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		model.add(Activation('sigmoid'))
		model.add(Dense(units=2, input_shape=(self.dataDim,), use_bias=True, kernel_initializer='normal'))
		model.add(Activation('softmax'))
		model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
		model.fit(self.trainX, self.trainY, epochs=self.num_epochs, batch_size=self.batch_size, verbose=1)
		evaluation = model.evaluate(self.testX, self.testY, verbose=1)
		preds = model.predict(self.testX, verbose=1)
		print("evaluation:",str(evaluation))
		print("preds:",str(preds))
		
		'''
		print("trainX:",str(self.trainX))
		print("trainY:",str(self.trainY))
		print("testX:",str(self.testX))
		print("testY:",str(self.testY))
		X = tf.placeholder("float", shape=[None, self.x_size])
		y = tf.placeholder("float", shape=[None, self.y_size])

		w_1 = self.init_weights((self.x_size, self.hidden_size))
		w_2 = self.init_weights((self.hidden_size, self.y_size))

		b_1 = self.init_weights((1, self.hidden_size))
		b_2 = self.init_weights((1, self.y_size))

		yhat = self.forwardprop(X, w_1, w_2, b_1, b_2)
		predict = tf.argmax(yhat, axis=1)

		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
		updates = tf.train.AdagradOptimizer(self.lr).minimize(cost)
		
		# train_batchX, train_batchY = tf.train.batch([self.trainX, self.trainY],batch_size=self.batch_size)
		# test_batchX, test_batchY = tf.train.batch([self.testX, self.testY],batch_size=self.batch_size)

		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)

		num_batches = math.ceil(self.trainingSize / self.batch_size)

		for epoch in range(self.num_epochs):
			for batch_num in range(num_batches):
				lb = batch_num * self.batch_size
				ub = min(lb + self.batch_size-1, self.trainingSize-1)
				sess.run(updates, feed_dict={X: self.trainX[lb:ub], y: self.trainY[lb:ub]})
			trainPreds = sess.run(predict, feed_dict={X: self.trainX, y: self.trainY})
			testPreds = sess.run(predict, feed_dict={X: self.testX, y: self.testY})

			#print("train acc:",str(self.getAccuracy(trainPreds, self.trainY)))
			print("* epoch:",str(epoch),"test acc:",str(self.getAccuracy(testPreds, self.testY)))
		'''
	def getAccuracy(self, preds, golds):
		return np.mean(np.argmax(golds, axis=1) == preds)

	def forwardprop(self, X, w_1, w_2, b_1, b_2):
		h1 = tf.nn.sigmoid(tf.add(tf.matmul(tf.nn.dropout(X, 1.0), w_1), b_1))
		return tf.add(tf.matmul(h1, w_2), b_2)

	def init_weights(self, shape):
		return tf.Variable(tf.random_normal(shape, stddev=0.1))

	def createData(self):
		# creates train data
		for i in range(self.trainingSize):
			curX = []
			curY = []
			if random.random() > 0.5: # generate positive
				for _ in range(self.dataDim):
					curX.append(random.uniform(0.66,1))
				curY = [0,1]
			else: # generate negative
				for _ in range(self.dataDim):
					curX.append(random.uniform(0.0,0.33))
				curY = [1,0]
			self.trainX.append(curX)
			self.trainY.append(curY)

		# creates test data
		for i in range(self.testingSize):
			curX = []
			curY = []
			if random.random() > 0.5: # generate positive
				for _ in range(self.dataDim):
					curX.append(random.uniform(0.66,1))
				curY = [0,1]
			else: # generate negative
				for _ in range(self.dataDim):
					curX.append(random.uniform(0.0,0.33))
				curY = [1,0]
			self.testX.append(curX)
			self.testY.append(curY)