try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import operator
import math
from collections import defaultdict
from get_coref_metrics import *
from random import random
from random import randint
class ECBHelper:

	def __init__(self, args, corpus): # goldTruthFile, goldLegendFile, isVerbose):
		#self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22]
		#self.devDirs = [23,24,25]
		self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20]
		self.devDirs = [21,22,23,24,25]
		self.testingDirs = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose
		self.args = args
	
		# filled in by loadEmbeddings(), if called, and if embeddingsType='type'
		self.wordTypeToEmbedding = {}
		#self.embeddingLength = 0 # filled in by loadEmbeddings()

		# filled in by addStanfordAnnotations(), if called
		self.posToIndex = {} # maps each of the 45 POS' to a unique index (alphabetical ordering), used for creating a feature
		self.badPOS = ["‘’", "``", "POS", "$", "''"]
		self.posToRandomEmbedding = {}
		self.posToGloveEmbedding = {}
		self.posEmbLength = 100

		# filled in by 
		self.wordToGloveEmbedding = {}
		self.wordEmbLength = 400

		# filled in by
		self.charToEmbedding = {}
		self.charEmbLength = 300

		# filled in by createSemantic*()
		self.SSMentionTypeToVec = {}
		self.SSEmbLength = -1

		# things are a little weird here; i wasn't originally loading the stopwords.
		# i only was within the SemanticSpace function.  but, now i want access to
		# stop words from the CCNN class, so i want to auto-load the stopwords, which i'm now doing
		self.stopWordsFile = self.args.stoplistFile
		self.stopwords = self.loadStopWords(self.stopWordsFile)

	def setValidDMs(self, DMs):
		self.validDMs = DMs
##################################################
#    creates DM pairs for train/dev/test
##################################################
##################################################
	'''
	def loadWordEmbeddings(self, embeddingsFile):
		print("self.args.lemmaType:",str(self.args.lemmaType))
		print("self.args.dependencyType:",str(self.args.dependencyType))
		if self.args.lemmaType == "none" and self.args.dependencyType == "none":
			return
		print("* in loadWordEmbeddings")
		self.wordToGloveEmbedding = {}
		f = open(embeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			word = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.wordToGloveEmbedding[word] = emb
		f.close()
	'''
	def loadStopWords(self, stopWordsFile):
		f = open(stopWordsFile, 'r')
		stopwords = set()
		for line in f:
			stopwords.add(line.rstrip().lower())
		f.close()
		return stopwords

	def createSemanticSpaceSimVectors(self, hddcrp_parsed):
		self.SSMentionTypeToVec = {}
		if self.args.SSType == "none":
			return
		print("* in createSemanticSpaceSimVectors()")
		W = self.args.SSwindowSize

		# stores mentions' types
		mentionTypes = set()
		stopwords = self.loadStopWords(self.stopWordsFile)

		# makes a set of all mentions Tokens
		mentionTokens = set()
		mentionTokens.update(self.getECBMentionTokens(self.trainingDirs))
		mentionTokens.update(self.getECBMentionTokens(self.devDirs))
		mentionTokens.update(self.getHDDCRPMentionTokens(hddcrp_parsed))
		for t in mentionTokens:
			if t.text not in stopwords and len(t.text) > 1:
				mentionTypes.add(t.text)
		print("# unique mention Types:",str(len(mentionTypes)))

		# gets the most popular N words (N = args.SSvectorSize)
		# which aren't stopwords and are > 1 char in length
		wordCounts = defaultdict(int)
		for t in self.corpus.corpusTokens:
			if t.text not in stopwords and len(t.text) > 1:
				wordCounts[t.text] += 1
		# puts the top N words into a 'topWords'
		sorted_wordCounts = sorted(wordCounts.items(), key=operator.itemgetter(1), reverse=True)
		commonTypes = [x[0] for x in sorted_wordCounts][0:self.args.SSvectorSize]

		# frees memory
		wordCounts = None
		sorted_wordCounts = None

		mentionWordsCounts = defaultdict(int)
		commonWordsCounts = defaultdict(int)
		mentionAndCommonCounts = defaultdict(int)

		for doc in self.corpus.docToTokens:

			# stores locations of all tokens we care about (both common words and mention tokens)
			mentionWordsLocations = defaultdict(set)
			commonWordsLocations = defaultdict(set)
			
			for t in self.corpus.docToTokens[doc]:
				if t.text in commonTypes:
					commonWordsLocations[t.text].add(int(t.tokenID))
					commonWordsCounts[t.text] += 1
				if t.text in mentionTypes:
					mentionWordsLocations[t.text].add(int(t.tokenID))
					mentionWordsCounts[t.text] += 1
			for m in mentionWordsLocations:
				for l in mentionWordsLocations[m]:
					lower = l - W
					upper = l + W
					for c in commonWordsLocations.keys():
						if c != m:
							for l2 in commonWordsLocations[c]:
								if l2 >= lower and l2 <= upper:
									mentionAndCommonCounts[(m,c)] = mentionAndCommonCounts[(m,c)] + 1
		
		# pre-compute these, so that we can use them if we do the log probs of PMI
		commonWordProbs = {}
		paddingValue = 0.00001
		unionWords = set()
		for c in commonWordsCounts.keys():
			unionWords.add(c)
		print("# unionWords:",str(len(unionWords)))
		for m in mentionWordsCounts.keys():
			unionWords.add(m)
		print("# unionWords:",str(len(unionWords)))
		sumCounts = sum(commonWordsCounts.values())
		sumCounts += float(paddingValue)*len(commonWordsCounts)
		for m in mentionWordsCounts:
			if m not in commonWordsCounts:
				sumCounts += mentionWordsCounts[m] + paddingValue
		for c in commonWordsCounts:
			commonWordProbs[c] = float(commonWordsCounts[c]) / float(sumCounts)
		
		#for c in sorted(commonWordProbs.items(), key=operator.itemgetter(1), reverse=True):
		#	print(c)

		for m in mentionTypes:
			vec = []
			for c in commonTypes:

				if commonWordProbs[c] == 0:
					print("* ERROR: commonWordProbs[c] == 0")
					exit(1)

				# calculates PMI via log probs
				if self.args.SSlog:
					#print("log prob")
					cooccurCount = paddingValue
					if (m,c) in mentionAndCommonCounts.keys():
						cwc = commonWordsCounts[c]
						mc = mentionWordsCounts[m]
						# sanity check
						if cwc == 0 or mc == 0:
							print("* ERROR: counts are incorrect")
							exit(1)
						cooccurCount += mentionAndCommonCounts[(m,c)]
					if mentionWordsCounts[m] <= 0:
						print("* ERROR: somehow, mentionWordsCounts[m] <= 0")
						exit(1)
					cooccurProb = float(cooccurCount) / float(mentionWordsCounts[m])	
					vec.append(math.log(float(cooccurProb) / float(commonWordProbs[c])))
				else: # calculates raw freq counts
					#print("NOT log prob")
					cooccurCount = float(paddingValue)
					if (m,c) in mentionAndCommonCounts.keys():
						cwc = commonWordsCounts[c]
						mc = mentionWordsCounts[m]
						# sanity check
						if cwc == 0 or mc == 0:
							print("* ERROR: counts are incorrect")
							exit(1)
						cooccurCount += mentionAndCommonCounts[(m,c)]
					vec.append(float(cooccurCount) / (float(commonWordsCounts[c]) * float(mentionWordsCounts[m])))

			self.SSMentionTypeToVec[m] = vec
			self.SSEmbLength = len(vec)

	def loadCharacterEmbeddings(self, charEmbeddingsFile):
		print("self.args.charEmbeddingsFile:",str(charEmbeddingsFile))
		if self.args.charType == "none":
			return
		print("* in loadCharacterEmbeddings")
		self.charToEmbedding = {}
		f = open(self.args.charEmbeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			char = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.charEmbLength = len(emb)
			self.charToEmbedding[char] = emb
		f.close()

	def loadPOSEmbeddings(self, embeddingsFile):
		self.posToGloveEmbedding = {}
		print("self.args.featurePOS:",str(self.args.featurePOS))
		if self.args.featurePOS == "none":
			return
		print("* in loadPOSEmbeddings")

		f = open(embeddingsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			pos = tokens[0]
			emb = [float(x) for x in tokens[1:]]
			self.posToGloveEmbedding[pos] = emb
		f.close()

	def constructECBDev(self, dirs):
		print("* in constructECBDev()")
		devTokenListPairs = []
		mentionIDPairs = []
		labels = []

		# finally, let's convert the DMs to their actual Tokens
		DMToTokenLists = {} # saves time
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum not in dirs:
				continue

			for doc_id in self.corpus.dirToDocs[dirNum]:
				docDMs = []
				for ref in self.corpus.docToREFs[doc_id]:
					for dm in self.corpus.docREFsToDMs[(doc_id,ref)]:
						if dm not in docDMs:
							docDMs.append(dm)

				added = set()
				for dm1 in docDMs:
					for dm2 in docDMs:
						if dm1 == dm2 or (dm1,dm2) in added or (dm2,dm1) in added:
							continue

						# sets dm1's
						tokenList1 = []
						if dm1 in DMToTokenLists.keys():
							tokenList1 = DMToTokenLists[dm1]
						else:
							tokenList1 = self.corpus.dmToMention[dm1].tokens
							DMToTokenLists[dm1] = tokenList1
						
						# sets dm2's
						tokenList2 = []
						if dm2 in DMToTokenLists.keys():
							tokenList2 = DMToTokenLists[dm2]
						else:
							tokenList2 = self.corpus.dmToMention[dm2].tokens
							DMToTokenLists[dm2] = tokenList2

						devTokenListPairs.append((tokenList1,tokenList2))
						mentionIDPairs.append((dm1,dm2))
						if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]:
							labels.append(1)
						else:
							labels.append(0)

						added.add((dm1,dm2))
						added.add((dm2,dm1))
		return (devTokenListPairs,mentionIDPairs,labels)

	# returns the Tokens (belonging to Mentions) within the hddcrp_parsed file
	def getHDDCRPMentionTokens(self, hddcrp_parsed):
		mentionTokens = set()
		for doc_id in hddcrp_parsed.docToHMentions.keys():
			for hm in hddcrp_parsed.docToHMentions[doc_id]:
				for t in hm.tokens:
					token = self.corpus.UIDToToken[t.UID] # does the linking b/w HDDCRP's parse and regular corpus
					mentionTokens.add(token)
		return mentionTokens

	# returns the Tokens (belonging to Mentions) within the passed-in dirs
	def getECBMentionTokens(self, dirs):
		mentionTokens = set()
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum not in dirs:
				continue
			for doc_id in self.corpus.dirToDocs[dirNum]:
				for dm in self.corpus.docToDMs[doc_id]:
					for t in self.corpus.dmToMention[dm].tokens:
						mentionTokens.add(t)
		return mentionTokens


	# constructs pairs of tokens and the corresponding labels, used for testing
	# RETURNS: (tokenListPairs, label)
	# e.g., (([Token1,Token2],[Token61]),1)
	def constructECBTraining(self, dirs):
		print("* in constructECBTraining()")
		trainingPositives = []
		trainingNegatives = []

		for dirNum in sorted(self.corpus.dirToREFs.keys()):

		    # only process the training dirs
			if dirNum not in dirs:
				continue

			added = set() # so we don't add the same pair twice
			for doc_id in self.corpus.dirToDocs[dirNum]:
				numRefsForThisDoc = len(self.corpus.docToREFs[doc_id])
				for i in range(numRefsForThisDoc):
					ref1 = self.corpus.docToREFs[doc_id][i]
					for dm1 in self.corpus.docREFsToDMs[(doc_id,ref1)]:
						for dm2 in self.corpus.docREFsToDMs[(doc_id,ref1)]:
							if dm1 != dm2 and (dm1,dm2) not in added and (dm2,dm1) not in added:

								# adds a positive example
								trainingPositives.append((dm1,dm2))
								added.add((dm1,dm2))
								added.add((dm2,dm1))
								numNegsAdded = 0
								j = i + 1
								while numNegsAdded < self.args.numNegPerPos:

									# pick the next REF
									ref2 = self.corpus.docToREFs[doc_id][j%numRefsForThisDoc]
									if numRefsForThisDoc == 1:
										doc_id2 = doc_id
										while doc_id2 == doc_id:
											numDocsInDir = len(self.corpus.dirToDocs[dirNum])
											doc_id2 = self.corpus.dirToDocs[dirNum][randint(0,numDocsInDir-1)]
										numRefsForDoc2 = len(self.corpus.docToREFs[doc_id2])
										ref2 = self.corpus.docToREFs[doc_id2][j%numRefsForDoc2]

										numDMs = len(self.corpus.docREFsToDMs[(doc_id2,ref2)])

										# pick a random negative from a different REF and different DOC
										dm3 = self.corpus.docREFsToDMs[(doc_id2,ref2)][randint(0, numDMs-1)]
										trainingNegatives.append((dm1,dm3))
										numNegsAdded += 1

									elif ref2 != ref1:
										numDMs = len(self.corpus.docREFsToDMs[(doc_id,ref2)])

										# pick a random negative from the different REF
										dm3 = self.corpus.docREFsToDMs[(doc_id,ref2)][randint(0, numDMs-1)]
										trainingNegatives.append((dm1,dm3))
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

		trainingTokenListPairs = []
		mentionIDPairs = []
		trainingLabels = []
		j = 0

		# finally, let's convert the DMs to their actual Tokens
		DMToTokenLists = {} # saves time
		for i in range(len(trainingPositives)):

			(dm1,dm2) = trainingPositives[i]			
			
			# sets dm1's
			tokenList1 = []
			if dm1 in DMToTokenLists.keys():
				tokenList1 = DMToTokenLists[dm1]
			else:
				tokenList1 = self.corpus.dmToMention[dm1].tokens
				DMToTokenLists[dm1] = tokenList1
			# sets dm2's
			tokenList2 = []
			if dm2 in DMToTokenLists.keys():
				tokenList2 = DMToTokenLists[dm2]
			else:
				tokenList2 = self.corpus.dmToMention[dm2].tokens
				DMToTokenLists[dm2] = tokenList2

			trainingTokenListPairs.append((tokenList1,tokenList2))
			mentionIDPairs.append((dm1,dm2))
			trainingLabels.append(1)

			# adds the negatives
			for _ in range(self.args.numNegPerPos):

				(dmneg1,dmneg2) = trainingNegatives[j]

				# first neg dm
				tokenListNeg1 = []
				if dmneg1 in DMToTokenLists.keys():
					tokenListNeg1 = DMToTokenLists[dmneg1]
				else:
					tokenListNeg1 = self.corpus.dmToMention[dmneg1].tokens
					DMToTokenLists[dmneg1] = tokenListNeg1

				# second neg dm
				tokenListNeg2 = []
				if dmneg2 in DMToTokenLists.keys():
					tokenListNeg2 = DMToTokenLists[dmneg2]
				else:
					tokenListNeg2 = self.corpus.dmToMention[dmneg2].tokens
					DMToTokenLists[dmneg2] = tokenListNeg2

				trainingTokenListPairs.append((tokenListNeg1,tokenListNeg2))
				mentionIDPairs.append((dmneg1,dmneg2))
				trainingLabels.append(0)
				j+=1
		return (trainingTokenListPairs,mentionIDPairs,trainingLabels)
	

	# creates all HM (DM equivalent) for test set (could be gold hmentions or predicted hmentions)
	def constructHDDCRPTest(self, hddcrp_parsed):
		hTokenListPairs = []
		mentionIDPairs = []
		labels = []
		numSingletons = 0
		for doc_id in hddcrp_parsed.docToHMentions.keys():
			HM_IDToTokenLists = {} # saves time
			added = set()

			if len(hddcrp_parsed.docToHMentions[doc_id]) == 1:
				print("*** :",str(doc_id),"has exactly 1 hmention")
				numSingletons += 1
				hm1 = hddcrp_parsed.docToHMentions[doc_id][0]
				tokenList = []
				for t in hm1.tokens:
					token = self.corpus.UIDToToken[t.UID] # does the linking b/w HDDCRP's parse and regular corpus
					tokenList.append(token)
				hTokenListPairs.append((tokenList,tokenList))
				mentionIDPairs.append((hm1.hm_id,hm1.hm_id))
				labels.append(1)
			else:
				for hm1 in hddcrp_parsed.docToHMentions[doc_id]:
					hm1_id = hm1.hm_id
					
					tokenList1 = []
					if hm1_id in HM_IDToTokenLists.keys():
						tokenList1 = HM_IDToTokenLists[hm1_id]
					else:
						for t in hm1.tokens:
							token = self.corpus.UIDToToken[t.UID] # does the linking b/w HDDCRP's parse and regular corpus
							tokenList1.append(token)
						HM_IDToTokenLists[hm1_id] = tokenList1

					for hm2 in hddcrp_parsed.docToHMentions[doc_id]:
						hm2_id = hm2.hm_id
						
						tokenList2 = []
						if hm2_id in HM_IDToTokenLists.keys():
							tokenList2 = HM_IDToTokenLists[hm2_id]
						else:
							for t in hm2.tokens:
								token = self.corpus.UIDToToken[t.UID] # does the linking b/w HDDCRP's parse and regular corpus
								tokenList2.append(token)
							HM_IDToTokenLists[hm2_id] = tokenList2

						if hm1_id == hm2_id or (hm1_id,hm2_id) in added or (hm2_id,hm1_id) in added:
							continue

						hTokenListPairs.append((tokenList1,tokenList2))
						mentionIDPairs.append((hm1_id,hm2_id))
						if hm1.ref_id == hm2.ref_id:
							labels.append(1)
						else:
							labels.append(0)
						added.add((hm1_id,hm2_id))
						added.add((hm2_id,hm1_id))
		return (hTokenListPairs,mentionIDPairs,labels)

	# creates all HM (DM equivalent) for test set
	def constructAllWDHMPairs(self, hddcrp_pred):
		pairs = []
		labels = []
		numSingletons = 0
		for doc_id in hddcrp_pred.docToHMentions.keys():
			added = set()
			if len(hddcrp_pred.docToHMentions[doc_id]) == 1:
				print("*** :",str(doc_id),"has exactly 1 hmention")
				numSingletons += 1
				hm1_id = hddcrp_pred.docToHMentions[doc_id][0].hm_id
				pairs.append((hm1_id,hm1_id))
				labels.append(1)
			else:
				for hm1 in hddcrp_pred.docToHMentions[doc_id]:
					hm1_id = hm1.hm_id
					for hm2 in hddcrp_pred.docToHMentions[doc_id]:
						hm2_id = hm2.hm_id
						if hm1_id == hm2_id or (hm1_id,hm2_id) in added or (hm2_id,hm1_id) in added:
							continue
						pairs.append((hm1_id,hm2_id))
						if hm1.ref_id == hm2.ref_id:
							labels.append(1)
						else:
							labels.append(0)
						added.add((hm1_id,hm2_id))
						added.add((hm2_id,hm1_id))
		return (pairs, labels)

## WITHIN-DOC
	'''
	def constructAllWDDMPairs(self, dirs):
		pairs = []
		labels = []
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum not in dirs:
				continue

			for doc_id in self.corpus.dirToDocs[dirNum]:
				docDMs = []
				for ref in self.corpus.docToREFs[doc_id]:
					for dm in self.corpus.docREFsToDMs[(doc_id,ref)]:
						if dm not in docDMs:
							docDMs.append(dm)

				added = set()
				for dm1 in docDMs:
					for dm2 in docDMs:
						if dm1 == dm2 or (dm1,dm2) in added or (dm2,dm1) in added:
							continue

						pairs.append((dm1,dm2))
						if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]:
							labels.append(1)
						else:
							labels.append(0)

						added.add((dm1,dm2))
						added.add((dm2,dm1))
		return (pairs, labels)
	'''
	'''
	def constructSubsampledWDDMPairs(self, dirs):
		print("* in constructSubsampledWDDMPairs()")
		trainingPositives = []
		trainingNegatives = []

		for dirNum in sorted(self.corpus.dirToREFs.keys()):

		    # only process the training dirs
			if dirNum not in dirs:
				continue

			added = set() # so we don't add the same pair twice
			for doc_id in self.corpus.dirToDocs[dirNum]:
				numRefsForThisDoc = len(self.corpus.docToREFs[doc_id])
				for i in range(numRefsForThisDoc):
					ref1 = self.corpus.docToREFs[doc_id][i]
					for dm1 in self.corpus.docREFsToDMs[(doc_id,ref1)]:
						for dm2 in self.corpus.docREFsToDMs[(doc_id,ref1)]:
							if dm1 != dm2 and (dm1,dm2) not in added and (dm2,dm1) not in added:

								# adds a positive example
								trainingPositives.append((dm1,dm2))
								added.add((dm1,dm2))
								added.add((dm2,dm1))
								numNegsAdded = 0
								j = i + 1
								while numNegsAdded < self.args.numNegPerPos:

									# pick the next REF
									ref2 = self.corpus.docToREFs[doc_id][j%numRefsForThisDoc]
									if numRefsForThisDoc == 1:
										doc_id2 = doc_id
										while doc_id2 == doc_id:
											numDocsInDir = len(self.corpus.dirToDocs[dirNum])
											doc_id2 = self.corpus.dirToDocs[dirNum][randint(0,numDocsInDir-1)]
										numRefsForDoc2 = len(self.corpus.docToREFs[doc_id2])
										ref2 = self.corpus.docToREFs[doc_id2][j%numRefsForDoc2]

										numDMs = len(self.corpus.docREFsToDMs[(doc_id2,ref2)])

										# pick a random negative from a different REF and different DOC
										dm3 = self.corpus.docREFsToDMs[(doc_id2,ref2)][randint(0, numDMs-1)]
										trainingNegatives.append((dm1,dm3))
										numNegsAdded += 1

									elif ref2 != ref1:
										numDMs = len(self.corpus.docREFsToDMs[(doc_id,ref2)])

										# pick a random negative from the different REF
										dm3 = self.corpus.docREFsToDMs[(doc_id,ref2)][randint(0, numDMs-1)]
										trainingNegatives.append((dm1,dm3))
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
	'''
#### CROSS-DOC (aka all pairs) #####
	def constructAllCDDMPairs(self, dirs):
		print("* in constructAllDMPairs()")
		pairs = []
		labels = []
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum not in dirs:
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

					pairs.append((dm1,dm2))
					if self.corpus.dmToREF[dm1] == self.corpus.dmToREF[dm2]:
						labels.append(1)
					else:
						labels.append(0)

					added.add((dm1,dm2))
					added.add((dm2,dm1))
		return (pairs, labels)

	def constructSubsampledCDDMPairs(self, dirs):
		print("* in constructSubsampledDMPairs()")
		trainingPositives = []
		trainingNegatives = []

		for dirNum in sorted(self.corpus.dirToREFs.keys()):

			# only process the training dirs
			if dirNum not in dirs:
				continue

			added = set() # so we don't add the same pair twice

			numRefsForThisDir = len(self.corpus.dirToREFs[dirNum]) 
			for i in range(numRefsForThisDir):
				ref1 = self.corpus.dirToREFs[dirNum][i]
				for dm1 in self.corpus.refToDMs[ref1]:
					for dm2 in self.corpus.refToDMs[ref1]:
						if dm1 != dm2 and (dm1,dm2) not in added and (dm2,dm1) not in added:
							# adds a positive example
							trainingPositives.append((dm1,dm2))
							added.add((dm1,dm2))
							added.add((dm2,dm1))

							numNegsAdded = 0
							j = i + 1
							while numNegsAdded < self.args.numNegPerPos:

								# pick the next REF
								ref2 = self.corpus.dirToREFs[dirNum][j%numRefsForThisDir]
								if ref2 == ref1:
									j += 1
									continue
								numDMs = len(self.corpus.refToDMs[ref2])

								# pick a random negative from the non-same REF
								dm3 = self.corpus.refToDMs[ref2][randint(0, numDMs-1)]
								trainingNegatives.append((dm1,dm3))
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

##################################################
#     CoNLL output files
##################################################
##################################################
	# returns a hashmap of clusters (sets)
	def constructCoNLLClustersFromFile(self, responseFile):
		ret = defaultdict(set)
		f = open(responseFile, 'r')
		f.readline()
		for line in f:
			line = line.rstrip()
			if line == "#end document":
				break
			_, dm, clusterID = line.rstrip().split()
			clusterID = clusterID[1:-1]
			ret[clusterID].add(dm)
			#if clusterID in clusterToDMs.keys():
			#	clusterToDMs[clusterID].add(dm)
			#else:
			#	tmp = set()
			#	tmp.add(dm)
			#	clusterToDMs[clusterID] = tmp
		return ret

	# constructs Truth WD file; used for evaluating via the CoNLL scorer.pl
	def writeCoNLLTruthFileWD(self, outputFile):
		f = open(outputFile, 'w')
		f.write("#begin document (t);\n")
		refNum = 0
		for d in self.corpus.dirToREFs:
			if d < 26:
				continue
			for ref in self.corpus.dirToREFs[d]:
				docsFoundSoFar = {}
				for dm in self.corpus.refToDMs[ref]:
					m = self.corpus.dmToMention[dm]
					if m.doc_id not in docsFoundSoFar.keys():
						docsFoundSoFar[m.doc_id] = refNum + 1
						refNum += 1

					clusterNum = docsFoundSoFar[m.doc_id]
					f.write(str(m.dirNum) + "\t" + str(m.doc_id) + ";" + \
						str(m.m_id) + "\t(" + str(clusterNum) + ")\n")
		f.write("#end document\n")
		f.close()

	# constructs Truth CD file; used for evaluating via the CoNLL scorer.pl
	def writeCoNLLTruthFileCD(self, outputFile):
		f = open(outputFile, 'w')
		f.write("#begin document (t);\n")
		refNum = 0
		for d in self.corpus.dirToREFs:
			if d < 26:
				continue
			for ref in self.corpus.dirToREFs[d]:
				for dm in self.corpus.refToDMs[ref]:
					m = self.corpus.dmToMention[dm]
					f.write(str(m.dirNum) + "\t" + str(m.doc_id) + ";" + \
						str(m.m_id) + "\t(" + str(refNum) + ")\n")
				refNum += 1
		f.write("#end document\n")
		f.close()

##################################################
##################################################

	def writeAllPOSToFile(self, outputFile):
		fout = open(outputFile, 'w')

		# for:
		# - word
		# - word's lemma
		# - dependency parent
		# - dependency parent's lemma
		# - dependency child
		# - dependency child's lemma
		allWordTypes = set()
		print("writing:",str(outputFile))
		for doc_id in self.corpus.docToGlobalSentenceNums.keys():
			for sent_num in sorted(self.corpus.docToGlobalSentenceNums[doc_id]):
				outLine = ""
				for t in self.corpus.globalSentenceNumToTokens[sent_num]:

					bestStanToken = self.getBestStanToken(t.stanTokens)

					pos = bestStanToken.pos
					allWordTypes.add(t.text)

					allWordTypes.add(bestStanToken.text)
					allWordTypes.add(bestStanToken.lemma)

					# look at its dependency parent
					for stanParentLink in bestStanToken.parentLinks:
						allWordTypes.add(stanParentLink.parent.lemma)
						allWordTypes.add(stanParentLink.parent.text)
					for stanChildLink in bestStanToken.childLinks:
						allWordTypes.add(stanChildLink.child.lemma)
						allWordTypes.add(stanChildLink.child.text)				#print("* we have:",str(len(bestStanToken.parentLinks)),"parents")
					#if len(bestStanToken.childLinks) != 1:
					#print("* we have:",str(len(bestStanToken.childLinks)),"children")

					#outLine += lemma + " "#pos + " "

		for t in allWordTypes:
			if t.startswith("'") or t.startswith("\""):
				print(str(t),"->",str(self.removeQuotes(t)))
			t = self.removeQuotes(t)
			if len(t) > 0:
				fout.write(str(t) + "\n")
		fout.close()

	# removes the leading and trailing quotes, if they exist
	def removeQuotes(self, token):
		if len(token) > 0:
			if token == "''" or token == "\"":
				return "\""
			elif token == "'" or token == "'s":
				return token
			else: # there's more substance to it, not a lone quote
				if token[0] == "'" or token[0] == "\"":
					token = token[1:]
				if len(token) > 0:
					if token[-1] == "'" or token[-1] == "\"":
						token = token[0:-1]
				return token
		else:
			print("* found a blank")
			return ""
	def getBestStanToken(self, stanTokens, token=None):
		longestToken = ""
		bestStanToken = None
		for stanToken in stanTokens:
			if stanToken.pos in self.badPOS:
				# only use the badPOS if no others have been set
				if bestStanToken == None:
					bestStanToken = stanToken
			else: # save the longest, nonBad POS tag
				if len(stanToken.text) > len(longestToken):
					longestToken = stanToken.text
					bestStanToken = stanToken
		if len(stanTokens) > 1 and token != None:
			print("token:",str(token.text),"=>",str(bestStanToken))

		if bestStanToken == None:
			print("* ERROR: our bestStanToken is empty!")
			exit(1)
		
		return bestStanToken

	# outputs our ECB corpus in plain-text format; 1 doc per doc, and 1 sentence per line
	def writeAllSentencesToFile(self, outputDir):
		
		for doc_id in self.corpus.docToGlobalSentenceNums.keys():
			fileOut = str(outputDir) + str(doc_id)
			fout = open(fileOut, 'w')
			print("(writing) :", str(fileOut))
			for sent_num in sorted(self.corpus.docToGlobalSentenceNums[doc_id]):
				outLine = ""
				for t in self.corpus.globalSentenceNumToTokens[sent_num]:
					outLine += t.text + " "
				outLine = outLine.rstrip()
				fout.write(outLine + "\n")
			fout.close()

	'''
	def X():
		pos = set()
		posTotal = set()
		longest = -1
		lengths = []
		for d in ourDocSet:
			for t in self.corpus.docToTokens[d]:
				tmp = ""
				for s in t.stanTokens:
					pos.add(s.pos)
					tmp += s.pos + "||"

					l = len(s.lemma)
					lengths.append(l)
					if l > longest:
						longest = l
				posTotal.add(tmp)
		for p in sorted(pos):
			print(p)
		print("we have pos:",str(len(pos)))
		print("# unique total:",str(len(posTotal)))
		print("longest lemma:",str(longest))
		print("avg:",str(sum(lengths)/float(len(lengths))))

		print("most sents:",str(mostNumSents))
		print("avg sent lengths:",str(sum(sentLengths)/float(len(sentLengths))))
	'''

	def addStanfordAnnotations(self, stanfordParser):

		# SANITY CHECK: ensures we're using the same, complete doc sets
		stanDocSet = set()
		ourDocSet = set()
		mostNumSents = 0
		sentLengths = []
		for doc_id in stanfordParser.docToSentenceTokens.keys():
			stanDocSet.add(doc_id)
		for doc_id in self.corpus.docToGlobalSentenceNums.keys():
			ourDocSet.add(doc_id)
			if doc_id not in stanDocSet:
				print("* ERROR: ecb has a doc:",str(doc_id)," which stan didn't parse")
				exit(1)
		for doc_id in stanDocSet:
			if doc_id not in ourDocSet:
				print("* ERROR: stan has a doc:",str(doc_id)," which ecb didn't parse")
				exit(1)
		# END OF SANITY CHECK

		# adds stan links on a per doc basis
		for doc_id in stanDocSet:
			#print("doc_id:",str(doc_id))

			# builds list of stanford tokens
			stanTokens = []
			if len(sorted(stanfordParser.docToSentenceTokens[doc_id].keys())) > mostNumSents:
				mostNumSents = len(sorted(stanfordParser.docToSentenceTokens[doc_id].keys()))
			sentLengths.append(len(sorted(stanfordParser.docToSentenceTokens[doc_id].keys())))

			for sent_num in sorted(stanfordParser.docToSentenceTokens[doc_id].keys()):
				for token_num in stanfordParser.docToSentenceTokens[doc_id][sent_num]:
					sToken = stanfordParser.docToSentenceTokens[doc_id][sent_num][token_num]
					if sToken.isRoot == False:
						stanTokens.append(sToken)
			
			ourTokens = [] # for readability,
			for sent_num in sorted(self.corpus.docToGlobalSentenceNums[doc_id]):
				for token in self.corpus.globalSentenceNumToTokens[sent_num]:
					ourTokens.append(token)
			#print("len(ourTokens):",str(len(ourTokens)))
			#print("len(self.corpus.docToTokens[doc_id]):",str(len(self.corpus.docToTokens[doc_id])))
			if len(ourTokens) != len(self.corpus.docToTokens[doc_id]):
				print("* ERROR: oddly, ourTokens (based on sentences) don't agree w/ our doc's tokens")
				'''
				for i in ourTokens:
					print(i)
				print("corpus docToTokens:")
				for i in self.corpus.docToTokens[doc_id]:
					print(i)
				'''	
				exit(1)

			j = 0
			i = 0
			while i < len(ourTokens):
				if j >= len(stanTokens):
					if i == len(ourTokens) - 1 and stanTokens[-1].text == "...":
						tmp = [stanTokens[-1]]
						ourTokens[i].addStanTokens(tmp)
						break
					else:
						print("ran out of stan tokens")
						exit(1)

				stanToken = stanTokens[j]
				ourToken = ourTokens[i]

				curStanTokens = [stanToken]
				curOurTokens = [ourToken]

				stan = stanToken.text
				ours = ourToken.text

				# pre-processing fixes since the replacements file can't handle spaces
				if stan == "''":
					stan = "\""
				elif stan == "2 1/2":
					stan = "2 1/2"
				elif stan == "3 1/2":
					stan = "3 1/2"
				elif stan == "877 268 9324":
					stan = "8772689324"
				elif stan == "0845 125 2222":
					stan = "08451252222"
				elif stan == "0800 555 111":
					stan = "0800555111"
				elif stan == "0800 555111":
					stan = "0800555111"
				elif stan == "0845 125 222":
					stan = "0845125222"

				# get the words to equal lengths first
				while len(ours) != len(stan):
					while len(ours) > len(stan):
						#print("\tstan length is shorter:", str(ours)," vs:",str(stan)," stanlength:",str(len(stan)))
						if j+1 < len(stanTokens):

							if stanTokens[j+1].text == "''":
								stanTokens[j+1].text = "\""
								print("TRYING TO FIX THE UPCOMING STAN TOKEN!")
							stan += stanTokens[j+1].text
								
							curStanTokens.append(stanTokens[j+1])
							if stan == "71/2":
								stan = "7 ½"
							elif stan == "31/2":
								stan = "3½"
							j += 1
							#print("\tstan is now:", str(stan))
						else:
							print("\tran out of stanTokens")
							exit(1)

					while len(ours) < len(stan):
						#print("\tour length is shorter:",str(ours),"vs:",str(stan),"stanlength:",str(len(stan)))
						if i+1 < len(ourTokens):
							ours += ourTokens[i+1].text
							curOurTokens.append(ourTokens[i+1])
							if ours == "31/2":
								ours = "3 1/2"
							elif ours == "21/2":
								#print("converted to: 2 1/2")
								ours = "2 1/2"
							elif ours == "31/2-inch":
								ours = "3 1/2-inch"
							elif ours == "3 1/2":
								ours = "3 1/2"
							i += 1
							#print("\tours is now:", str(ours))
						else:
							print("\tran out of ourTokens")
							exit(1)	

				if ours != stan:
					print("\tMISMATCH: [",str(ours),"] [",str(stan),"]")
					'''
					for i in range(len(ours)):
						print(ours[i],stan[i])
						if ours[i] != stan[i]:
							print("those last ones didnt match!")
					'''
					exit(1)
				else: # texts are identical, so let's set the stanTokens
					for t in curOurTokens:
						t.addStanTokens(curStanTokens)

				'''
				if len(curStanTokens) > len(curOurTokens):
					numStanGreater += 1

					if len(curOurTokens) > 1:
						#print("**** more than 1 of our tokens map to more than 1 stan token")
						print("**** # stan tokens:",len(curStanTokens)," # our tokens:",len(curOurTokens))
						for s in curStanTokens:
							print("\t",str(s))
						for c in curOurTokens:
							print("\t",str(c))
					
				elif len(curStanTokens) < len(curOurTokens):
					
					
					print("# stan tokens:",len(curStanTokens)," # our tokens:",len(curOurTokens))
					for s in curStanTokens:
						print("\t",str(s))
					for c in curOurTokens:
						print("\t",str(c))
					numOursGreater += len(curOurTokens)
				else:
					numEqual += 1
				'''

				j += 1
				i += 1
			
			# ensures every Token in the doc has been assigned at least 1 StanToken
			for t in self.corpus.docToTokens[doc_id]:
				if len(t.stanTokens) == 0:
					print("Token:",str(t)," never linked w/ a stanToken!")
					exit(1)
		print("we've successfully added stanford links to every single token within our",str(len(ourDocSet)),"docs")
		
		# makes:
		# (1) a mapping from POS -> index
		# (2) random embedding for each POS
		i = 0
		for pos in sorted(stanfordParser.posTags):
			
			# does 1
			self.posToIndex[pos] = i
			i += 1			
			
			# does 2 (makes random embedding)
			randEmb = []
			for _ in range(self.posEmbLength):
				randEmb.append(random())
			self.posToRandomEmbedding[pos] = randEmb

		# following line should be commented out when we're creating a POS/LEMMA/etc file, before we have the embeddings
		self.loadPOSEmbeddings(self.args.posEmbeddingsFile)
		# self.loadLemmaEmbeddings(self.args.lemmaEmbeddingsFile)
		self.loadCharacterEmbeddings(self.args.charEmbeddingsFile)