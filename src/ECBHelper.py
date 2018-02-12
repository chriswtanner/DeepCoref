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
##pos: 2263
#neg: 4526
	def __init__(self, args, corpus, hddcrp_parsed, runFFNN):

		self.useDoubleDevDirs = True # should only be True when using ECBTest w/ FFNN
		self.onlyCrossDoc = False # only relevant if we are doing CD, in which case True = dont use WD pairs.  False = use all WD and CD pairs
		self.nonTestingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25]
		self.trainingDirs = []
		self.devDirs = []
		for _ in self.nonTestingDirs:
			if _ >= args.devDir: # e.g., pass in "23" if you want the dev dirs to be 23-25
				self.devDirs.append(_)
			else:
				self.trainingDirs.append(_)

		# if we passed in one of the k-fold cross-validate ones, then let's make dev = all, training = none
		if len(self.devDirs) == 0:
			self.devDirs = self.nonTestingDirs
			self.trainingDirs = []

		if self.useDoubleDevDirs and runFFNN: # if true, we will use some of the Training dirs as a separate, smaller Dev A set (20-22)
			self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19]
			self.devDirs = [20,21] # will serve as Training for FFNN
			self.testingDirs = [23,24] #,24,25]
		else:
			self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22]
			self.devDirs = [23,24,25] # will serve as Training for FFNN
			self.testingDirs = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]
		print("trainingDirs:",str(self.trainingDirs))
		print("devDirs:",str(self.devDirs))
		print("testingDirs:",str(self.testingDirs))

		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose
		self.args = args
		self.hddcrp_parsed = hddcrp_parsed

		# filled in by loadEmbeddings(), if called, and if embeddingsType='type'
		self.wordTypeToEmbedding = {}

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

		if self.args.SSType != "none":
			self.createSemanticSpaceSimVectors() # just uses args and corpus

	def setValidDMs(self, DMs):
		self.validDMs = DMs
##################################################
#    creates DM pairs for train/dev/test
##################################################
##################################################

	def loadStopWords(self, stopWordsFile):
		f = open(stopWordsFile, 'r')
		stopwords = set()
		for line in f:
			stopwords.add(line.rstrip().lower())
		f.close()
		return stopwords

	def createSemanticSpaceSimVectors(self):
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
		mentionTokens.update(self.getHDDCRPMentionTokens(self.hddcrp_parsed))
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

	def constructECBDev(self, dirs, isTest, isWDModel): # the 2nd-to-last param determines if we're working on ECBTest
		print("* in constructECBDev()")
		devTokenListPairs = []
		mentionIDPairs = []
		labels = []

		# finally, let's convert the DMs to their actual Tokens
		DMToTokenLists = {} # saves time
		for dirNum in sorted(self.corpus.dirToREFs.keys()):
			if dirNum not in dirs:
				continue

			if isWDModel:
				for doc_id in self.corpus.dirToDocs[dirNum]:
					docDMs = []
					for ref in self.corpus.docToREFs[doc_id]:
						for dm in self.corpus.docREFsToDMs[(doc_id,ref)]:
							if dm not in docDMs:
								docDMs.append(dm)

					# we only have 1 DM in the doc, but we want to add it if we're w/ the test set
					if isTest and len(docDMs) == 1:
						dm1 = docDMs[0]
						tokenList1 = self.corpus.dmToMention[dm1].tokens
						devTokenListPairs.append((tokenList1,tokenList1))
						mentionIDPairs.append((dm1,dm1))
						labels.append(1)
					else:
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
			else: # make CD pairs
				added = set()
				for doc_id1 in self.corpus.dirToDocs[dirNum]:
					docDMs1 = []
					for ref in self.corpus.docToREFs[doc_id1]:
						for dm in self.corpus.docREFsToDMs[(doc_id1,ref)]:
							if dm not in docDMs1:
								docDMs1.append(dm)
					for doc_id2 in self.corpus.dirToDocs[dirNum]:
						if self.onlyCrossDoc and doc_id1 == doc_id2:
							continue
						docDMs2 = []
						for ref in self.corpus.docToREFs[doc_id2]:
							for dm in self.corpus.docREFsToDMs[(doc_id2,ref)]:
								if dm not in docDMs2:
									docDMs2.append(dm)

						# iterate over all relevant dms, where they either came from same doc or not
						for dm1 in docDMs1:
							(doc_id1,m1) = dm1
							extension1 = doc_id1[doc_id1.find("ecb"):]
							for dm2 in docDMs2:
								(doc_id2,m2) = dm2
								extension2 = doc_id2[doc_id2.find("ecb"):]		
								if extension1 != extension2 or dm1 == dm2 or (dm1,dm2) in added or (dm2,dm1) in added:
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
	def constructECBTraining(self, dirs, isWDModel):
		print("* in constructECBTraining()")
		trainingPositives = []
		trainingNegatives = []

		for dirNum in sorted(self.corpus.dirToREFs.keys()):

			# only process the training dirs
			if dirNum not in dirs:
				continue

			added = set() # so we don't add the same pair twice

			if isWDModel:
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
			else: # generate CD pairs
				numRefsForThisDir = len(self.corpus.dirToREFs[dirNum])
				
				for i in range(numRefsForThisDir):
					ref1 = self.corpus.dirToREFs[dirNum][i]
					extensions = set() # keeps track of if the docs are ecb.xml or ecbplus.xml (to ensure we have homogenous pairs)
					for dm1 in self.corpus.refToDMs[ref1]:
						(doc_id1,m1) = dm1
						extension1 = doc_id1[doc_id1.find("ecb"):]
						for dm2 in self.corpus.refToDMs[ref1]:
							(doc_id2,m2) = dm2
							extension2 = doc_id2[doc_id2.find("ecb"):]

							# adds positive examples first
							if extension1 == extension2 and dm1 != dm2 and (dm1,dm2) not in added and (dm2,dm1) not in added:
								if self.onlyCrossDoc and doc_id1 == doc_id2:
									continue

								# adds a positive example
								trainingPositives.append((dm1,dm2))
								added.add((dm1,dm2))
								added.add((dm2,dm1))
								numNegsAdded = 0
								j = i + 1
								while numNegsAdded < self.args.numNegPerPos:
									# pick the next REF (which ~50% of the time will be in
									# the other half/extension of the dir)
									ref2 = self.corpus.dirToREFs[dirNum][j%numRefsForThisDir]
									if ref2 != ref1:

										# pick a random negative from the different REF
										numDMsForOtherRef = len(self.corpus.refToDMs[ref2])
										dm3 = self.corpus.refToDMs[ref2][randint(0,numDMsForOtherRef-1)]

										# but only add the negative example if it's in the same half/extension
										(doc_id3,m3) = dm3
										extension3 = doc_id3[doc_id3.find("ecb"):]
										if extension1 == extension3:
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
		'''
		for p in trainingPositives:
			(dm1,dm2) = p
			print(self.corpus.dmToMention[dm1]," and ",self.corpus.dmToMention[dm2])
		'''
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
	def constructHDDCRPTest(self, hddcrp_parsed, isWDModel):
		print("* in constructHDDCRPTest()")

		hTokenListPairs = []
		mentionIDPairs = []
		labels = []
		numSingletons = 0

		if isWDModel:
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
		else: # generate CD pairs
			for dir_num in hddcrp_parsed.dirToDocs:
				HM_IDToTokenLists = {} # saves time
				added = set()

				for doc_id1 in hddcrp_parsed.dirToDocs[dir_num]:
					extension1 = doc_id1[doc_id1.find("ecb"):]

					for doc_id2 in hddcrp_parsed.dirToDocs[dir_num]:
						extension2 = doc_id2[doc_id2.find("ecb"):]
						
						# if we want to exclude WD pairs, we must be in diff docs
						if self.onlyCrossDoc and doc_id1 == doc_id2:
							continue
						# docs must come from same dir half
						if extension1 != extension2:
							continue

						for hm1 in hddcrp_parsed.docToHMentions[doc_id1]:
							hm1_id = hm1.hm_id
							tokenList1 = []
							if hm1_id in HM_IDToTokenLists.keys():
								tokenList1 = HM_IDToTokenLists[hm1_id]
							else:
								for t in hm1.tokens:
									token = self.corpus.UIDToToken[t.UID] # does the linking b/w HDDCRP's parse and regular corpus
									tokenList1.append(token)
								HM_IDToTokenLists[hm1_id] = tokenList1
							for hm2 in hddcrp_parsed.docToHMentions[doc_id2]:
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

	# converts the WD file (either gold or predictions) to a CD format
	# NOTE: this should be run whenever we use a CD Model
	# that is, we first create a WD file just to ensure we're
	# output'ing each token in the correct format as the HDDCRP
	def convertWDFileToCDFile(self, stoppingPoint):
		# constructs output file
		fileBase = str(self.args.resultsDir) + \
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
			"co" + str(self.args.CCNNOpt) + "_" + \
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
			"dd" + str(self.args.devDir) + "_" + \
			"fn" + str(self.args.FFNNnumEpochs) + "_" + \
			"fp" + str(self.args.FFNNPosRatio) + "_" + \
			"fo" + str(self.args.FFNNOpt) + "_" + \
			"sp" + str(stoppingPoint)
		wdFile = fileBase + ".WD.txt"
		cdFile = fileBase + ".CD.txt"

		fin = open(wdFile, 'r')
		fout = open(cdFile, 'w')
		print("ECHelper writing out:",str(cdFile))
		dirHalfToLines = defaultdict(lambda : defaultdict(list))
		for line in fin:
			line = line.rstrip()
			if line.startswith("#") or line == "":
				continue
			doc_id = line.split("\t")[0]
			ext = doc_id[doc_id.find("ecb"):doc_id.find(".xml")]
			dir_num = doc_id.split("_")[0]
			key = str(dir_num) + "_" + str(ext)
			dirHalfToLines[dir_num][ext].append(key + line[line.find("\t"):])
		print("# dirs writing out:",str(len(dirHalfToLines.keys())))
		for dir_num in sorted(dirHalfToLines.keys()):
			for ext in sorted(dirHalfToLines[dir_num]):
				fout.write("#begin document (" + str(dir_num) + "_" + str(ext) + "); part 000\n")
				for l in dirHalfToLines[dir_num][ext]:
					fout.write(l + "\n")
				fout.write("\n#end document\n")
		fin.close()
		fout.close()

	# writes CoNLL file in the same format as args.hddcrpFile
	def writeCoNLLFile(self, predictedClusters, stoppingPoint):
		hm_idToClusterID = {}
		for c_id in predictedClusters.keys():
			for hm_id in predictedClusters[c_id]:
				hm_idToClusterID[hm_id] = c_id

		print("# hm_ids:",str(len(hm_idToClusterID.keys())))

		# sanity check
		'''
		for hm_id in self.hddcrp_parsed.hm_idToHMention.keys():
			if hm_id not in hm_idToClusterID.keys():
				print("ERROR: hm_id:",str(hm_id),"NOT FOUND within our clusters, but it's parsed!")
				exit(1)
		'''
		# constructs output file
		fileOut = str(self.args.resultsDir) + \
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
			"co" + str(self.args.CCNNOpt) + "_" + \
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
			"dd" + str(self.args.devDir) + "_" + \
			"fn" + str(self.args.FFNNnumEpochs) + "_" + \
			"fp" + str(self.args.FFNNPosRatio) + "_" + \
			"fo" + str(self.args.FFNNOpt) + "_" + \
			"sp" + str(stoppingPoint) + \
			".WD.txt"

		print("ECBHelper writing out:",str(fileOut))
		fout = open(fileOut, 'w')

		# reads the original CoNLL, while writing each line
		f = open(self.args.hddcrpFullFile, 'r')
		tokenIndex = 0
		REFToStartTuple = defaultdict(list)
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
				fout.write(line + "\n")
			elif line == "":
				sentenceNum += 1
				fout.write(line + "\n")
			elif len(tokens) == 5:
				doc, _, tokenNum, text, ref_ = tokens   
				UID = str(doc) + ";" + str(sentenceNum) + ";" + str(tokenNum)

				# reconstructs the HMention(s) that exist on this line, for the
				# sake of being able to now look up what cluster assignent it/they belong to
				htoken = self.hddcrp_parsed.UIDToToken[UID]
				hmentions = set()
				for hm_id in htoken.hm_ids:
					hmentions.add(self.hddcrp_parsed.hm_idToHMention[hm_id])

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 "|""
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				if (len(refs) == 1 and refs[0] == "-"):
					fout.write(line + "\n") # just output it, since we want to keep the same mention going
				else:
					ref_section = ""
					isFirst = True
					for ref in refs:
						if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
							ref_id = int(ref[1:])
							REFToStartTuple[ref_id].append((tokenIndex,isFirst))
							startTuple=(tokenIndex,isFirst)
							foundMention = False
							for hmention in hmentions:
								if hmention.ref_id == ref_id and hmention.startTuple == startTuple: # we found the exact mention
									foundMention = True
									hm_id = hmention.hm_id

									if hm_id in hm_idToClusterID:
										clusterID = hm_idToClusterID[hm_id]
										ref_section += "(" + str(clusterID)
										break
							if not foundMention:
								print("* ERROR #1, we never found the mention for this line:",str(line))
								ref_section = "-"
								#exit(1)

						# represents we are ending a mention
						elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
							ref_id = -1

							endTuple=(tokenIndex,isFirst)
							startTuple = ()
							# we set ref_if, tokens, UID
							if ref[0] != "(": # ref_id)
								ref_id = int(ref[:-1])
								startTuple = REFToStartTuple[ref_id].pop()
							else: # (ref_id)
								ref_id = int(ref[1:-1])
								startTuple = (tokenIndex,isFirst)
								ref_section += "("

							#print("starttuple:",str(startTuple))
							#print("endTuple:",str(endTuple))

							foundMention = False
							for hmention in hmentions:
								# print("looking at hmention:",str(hmention))
								if hmention.ref_id == ref_id and hmention.startTuple == startTuple and hmention.endTuple == endTuple: # we found the exact mention
									foundMention = True
									hm_id = hmention.hm_id
									if hm_id in hm_idToClusterID:
										clusterID = hm_idToClusterID[hm_id]
										ref_section += str(clusterID) + ")"
										break
							if not foundMention:
								print("* ERROR #2, we never found the mention for this line:",str(line))
								ref_section = "-"
								#exit(1)

						if len(refs) == 2 and isFirst:
							ref_section += "|"
						isFirst = False
					fout.write(str(doc) + "\t" + str(_) + "\t" + str(tokenNum) + \
						"\t" + str(text) + "\t" + str(ref_section) + "\n")
					# end of current token line
				tokenIndex += 1 # this always increases whenever we see a token

		f.close()
		fout.close()

	# creates clusters for our hddcrp predictions
	def clusterHPredictions(self, pairs, predictions, stoppingPoint, isWDModel):
		clusters = {}

		if isWDModel:
			print("in clusterHPredictions() -- WD Model")
			# stores predictions
			docToHMPredictions = defaultdict(lambda : defaultdict(float))
			docToHMs = defaultdict(list) # used for ensuring our predictions included ALL valid HMs
			
			uniqueHMs = set()
			for i in range(len(pairs)):
				(hm1,hm2) = pairs[i]

				prediction = predictions[i][0]
				doc_id = self.hddcrp_parsed.hm_idToHMention[hm1].doc_id
				doc_id2 = self.hddcrp_parsed.hm_idToHMention[hm2].doc_id
				if doc_id != doc_id2:
					print("ERROR: pairs are from diff docs")
					exit(1)

				if hm1 not in docToHMs[doc_id]:
					docToHMs[doc_id].append(hm1)
				if hm2 not in docToHMs[doc_id]:
					docToHMs[doc_id].append(hm2)

				docToHMPredictions[doc_id][(hm1,hm2)] = prediction
				uniqueHMs.add(hm1)
				uniqueHMs.add(hm2)
			
			ourClusterID = 0
			ourClusterSuperSet = {}
			for doc_id in docToHMPredictions.keys():
				# constructs our base clusters (singletons)
				ourDocClusters = {} 
				for i in range(len(docToHMs[doc_id])):
					hm = docToHMs[doc_id][i]
					a = set()
					a.add(hm)
					ourDocClusters[i] = a

				# the following keeps merging until our shortest distance > stopping threshold,
				# or we have 1 cluster, whichever happens first
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
										if (dm1,dm2) in docToHMPredictions[doc_id]:
											dist = docToHMPredictions[doc_id][(dm1,dm2)]
										elif (dm2,dm1) in docToHMPredictions[doc_id]:
											dist = docToHMPredictions[doc_id][(dm2,dm1)]
										else:
											print("* error, why don't we have either dm1 or dm2 in doc_id")
											exit(1)
										avgavgdists.append(dist)
										avgdists.append(dist)
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
					ourClusterID += 1
			# end of going through every doc
			#print("# our clusters:",str(len(ourClusterSuperSet)))
			return ourClusterSuperSet
		else: # CD model
			print("in clusterHPredictions() -- CD Model")
			# stores predictions
			dirHalfToHMPredictions = defaultdict(lambda : defaultdict(float))
			dirHalfToHMs = defaultdict(list) # used for ensuring our predictions included ALL valid HMs
			
			uniqueHMs = set()

			# separates the predictions into dir-halves, so that we
			# don't try to cluster mentions all in the same dir
			for i in range(len(pairs)):
				(hm1,hm2) = pairs[i]

				prediction = predictions[i][0]
				doc_id = self.hddcrp_parsed.hm_idToHMention[hm1].doc_id
				doc_id2 = self.hddcrp_parsed.hm_idToHMention[hm2].doc_id

				extension1 = doc_id[doc_id.find("ecb"):]
				dir_num1 = int(doc_id.split("_")[0])

				extension2 = doc_id2[doc_id2.find("ecb"):]
				dir_num2 = int(doc_id2.split("_")[0])			

				if dir_num1 != dir_num2:
					print("ERROR: pairs are from diff docs")
					exit(1)
				key1 = str(dir_num1) + extension1
				key2 = str(dir_num2) + extension2

				if key1 != key2:
					print("* ERROR, somehow, training pairs came from diff dir-halves")
					exit(1)

				if hm1 not in dirHalfToHMs[key1]:
					dirHalfToHMs[key1].append(hm1)
				if hm2 not in dirHalfToHMs[key2]:
					dirHalfToHMs[key2].append(hm2)

				dirHalfToHMPredictions[key1][(hm1,hm2)] = prediction
				uniqueHMs.add(hm1)
				uniqueHMs.add(hm2)
			
			ourClusterID = 0
			ourClusterSuperSet = {}
			for key in dirHalfToHMPredictions.keys():
				#print("key:",str(key))
				#print("# dirHalfToHMs:", str(len(dirHalfToHMs[key])))
				#print("they are:",str(dirHalfToHMs[key]))
				#print("dirHalfToHMPredictions:",str(len(dirHalfToHMPredictions[key])))
				#print("and they are:",str(dirHalfToHMPredictions[key]))
				# constructs our base clusters (singletons)
				ourDirHalfClusters = {}
				for i in range(len(dirHalfToHMs[key])):
					hm = dirHalfToHMs[key][i]
					a = set()
					a.add(hm)
					ourDirHalfClusters[i] = a

				#print("# dirHalfClusters:",str(len(ourDirHalfClusters.keys())))
				# the following keeps merging until our shortest distance > stopping threshold,
				# or we have 1 cluster, whichever happens first
				while len(ourDirHalfClusters.keys()) > 1:
					# find best merge
					closestDist = 999999
					closestClusterKeys = (-1,-1)

					closestAvgDist = 999999
					closestAvgClusterKeys = (-1,-1)

					closestAvgAvgDist = 999999
					closestAvgAvgClusterKeys = (-1,-1)

					# looks at all combinations of pairs
					# starting case, each c1/key/cluster has a single HM
					i = 0
					for c1 in ourDirHalfClusters.keys():
						j = 0
						for c2 in ourDirHalfClusters.keys():
							if j > i:
								avgavgdists = []
								for dm1 in ourDirHalfClusters[c1]:
									avgdists = []
									for dm2 in ourDirHalfClusters[c2]:
										dist = 99999
										#print("dm1:",str(dm1),"dm2:",str(dm2))
										if (dm1,dm2) in dirHalfToHMPredictions[key]:
											dist = dirHalfToHMPredictions[key][(dm1,dm2)]
										elif (dm2,dm1) in dirHalfToHMPredictions[key]:
											dist = dirHalfToHMPredictions[key][(dm2,dm1)]
										else:
											print("* error, why don't we have either (dm1,dm2) or (dm2,dm1) in key:",str(key))
											exit(1)
										#print("dist:",str(dist))
										avgavgdists.append(dist)
										avgdists.append(dist)
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

					for _ in ourDirHalfClusters[c1]:
						newCluster.add(_)
					for _ in ourDirHalfClusters[c2]:
						newCluster.add(_)
					ourDirHalfClusters.pop(c1, None)
					ourDirHalfClusters.pop(c2, None)
					ourDirHalfClusters[c1] = newCluster
				# end of clustering current dir-half
				for i in ourDirHalfClusters.keys():
					ourClusterSuperSet[ourClusterID] = ourDirHalfClusters[i]
					ourClusterID += 1
			# end of going through every dir-half
			print("# our clusters:",str(len(ourClusterSuperSet)))
			return ourClusterSuperSet
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