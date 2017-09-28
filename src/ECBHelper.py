try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class ECBHelper:

	def __init__(self, corpus, args): # goldTruthFile, goldLegendFile, isVerbose):
		
		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose
		self.args = args
	
		# filled in by loadEmbeddings(), if called, and if embeddingsType='type'
		self.wordTypeToEmbedding = {}

		# filled in by loadEmbeddings() (it makes a matrix, where # of rows is how many words to include)
		self.dmToEmbedding = {}
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
		exit(1)


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
			f.close()


	def createCCNNData(self):

		self.loadEmbeddings(self.args.embeddingsFile, self.args.embeddingsType)

		for m in self.corpus.mentions:
			print("mention:",str(m))
			t_startIndex = 99999999
			t_endIndex = -1
			for t in m.corpusTokenIndices:
				token = self.corpus.corpusTokens[t]
				ind = self.corpus.corpusTokensToCorpusIndex[token]
				if ind < t_startIndex:
					t_startIndex = ind
				if ind > t_endIndex:
					t_endIndex = ind

			# the prev tokens
			for i in range(self.args.windowSize):
				ind = t_startIndex - self.args.windowSize + i

				emd = [0]*50
				if ind >= 0:
					token = self.corpus.corpusTokens[ind]
					print("prev:",str(token))
					emd = self.wordTypeToEmbedding[token.text]

			for i in range(self.args.windowSize):
				ind = t_endIndex + 1 + i

				emd = [0] * 50
				if ind < self.corpus.numCorpusTokens - 1:
					token = self.corpus.corpusTokens[ind]
					print("next",str(token))
					emd = self.wordTypeToEmbedding[token.text]

		return

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