try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from random import randint
class ECBHelper:

	def __init__(self, corpus, args): # goldTruthFile, goldLegendFile, isVerbose):
		
		self.trainingCutoff = 25 # anything higher than this will be testing

		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose
		self.args = args
	
		# filled in by loadEmbeddings(), if called, and if embeddingsType='type'
		self.wordTypeToEmbedding = {}

		self.embeddingLength = 0 # filled in by loadEmbeddings()

		emb1 = [1]*50
		emb2 = [2]*50
		emb3 = [3]*50

		emb4 = [4]*50
		emb5 = [5]*50
		emb6 = [6]*50

		emb7 = [7]*50
		emb8 = [8]*50
		emb9 = [9]*50

		m1 = np.empty(shape=(3,50))
		m1[0] = emb1
		m1[1] = emb2
		m1[2] = emb3

		m2 = np.empty(shape=(3,50))
		m2[0] = emb4
		m2[1] = emb5
		m2[2] = emb6

		m3 = np.empty(shape=(3,50))
		m3[0] = emb7
		m3[1] = emb8
		m3[2] = emb9



		m1 = np.asarray(m1).reshape(3,50,1)
		m2 = np.asarray(m2).reshape(3,50,1)
		m3 = np.asarray(m3).reshape(3,50,1)
		pair1 = np.asarray([m1,m2])
		pair2 = np.asarray([m1,m3])
		pair3 = np.asarray([m2,m3])
		pairs = []
		pairs.append(pair1)
		pairs.append(pair2)
		pairs.append(pair3)
		print(np.asarray(pairs).shape)
		print(pairs[0])

	def constructTestingPairs(self):
		testingPairs = []
		testingLabels = []
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum <= self.trainingCutoff:
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

					testingPairs.append((dm1,dm2))
					if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]:
						testingLabels.append(1)
					else:
						testingLabels.append(0)	

					added.add((dm1,dm2))
					added.add((dm2,dm1))
		return (testingPairs, testingLabels)

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

	def createCCNNData(self):

		# loads embeddings
		self.loadEmbeddings(self.args.embeddingsFile, self.args.embeddingsType)

		# loads the list of DM pairs we'll care about constructing
		(trainingPairs, trainingLabels) = self.constructTrainingPairs()
		(testingPairs, testingLabels) = self.constructTestingPairs()

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

		# constructs final training 5D matrix
		train_X = []
		for (dm1,dm2) in trainingPairs:
			pair = np.asarray([dmToMatrix[dm1],dmToMatrix[dm2]])
			train_X.append(pair)
		train_Y = np.asarray(trainingLabels)
		train_X = np.asarray(train_X)

		# constructs final testing 5D matrix
		test_X = []
		for (dm1,dm2) in testingPairs:
			pair = np.asarray([dmToMatrix[dm1],dmToMatrix[dm2]])
			test_X.append(pair)
		test_Y = np.asarray(testingLabels)
		test_X = np.asarray(test_X)

		return ((train_X,train_Y),(test_X, test_Y))

	# iterates through the corpus, printing 1 sentence per line
	def writeAllSentencesToFile(self, outputFile):
		fout = open(outputFile, 'w')
		lasts = set()
		for sent_num in sorted(self.corpus.globalSentenceNumToTokens.keys()):
			outLine = ""
			print("(writing) sent_num:", str(sent_num))
			#if sent_num > 5000:
			#	break

			lastToken = ""
			for t in self.corpus.globalSentenceNumToTokens[sent_num]:
				outLine += t.text + " "
				lastToken = t.text

			lasts.add(lastToken)
			outLine = outLine.rstrip()
			fout.write(outLine + "\n")
		fout.close()

		for i in lasts:
			print(str(i))

	# sentences
	#	sentence #1 of 17
	#		tokens
	#			token 1 of 28
	#				word
	#				lemma
	#				characteroffsetbegin
	#				characteroffsetend
	#				pos
	#				ner
	#				speaker
	#	sentence #2 of 17
	# coreference

		'''
		tree = ET.parse(stanFile)
		root = tree.getroot()
		for sentence in ET.find('sentence'):

			sentence_id = sentence.get('id')	
			print sentence_id
		'''