try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from get_coref_metrics import *
from random import randint
class ECBHelper:

	def __init__(self, args, corpus): # goldTruthFile, goldLegendFile, isVerbose):
		self.trainingDirs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22]
		self.devDirs = [23,24,25]
		self.testingDirs = [26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45]

		self.trainingCutoff = 25 # anything higher than this will be testing

		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose
		self.args = args
	
		# filled in by loadEmbeddings(), if called, and if embeddingsType='type'
		self.wordTypeToEmbedding = {}

		self.embeddingLength = 0 # filled in by loadEmbeddings()


	def setValidDMs(self, DMs):
		self.validDMs = DMs

##################################################
#    creates DM pairs for train/dev/test
##################################################
##################################################

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

	def addStanfordAnnotations(self, stanfordParser):
		for doc_id in stanfordParser.docToSentenceTokens.keys():
			print("doc_id:",str(doc_id))
			print("# stan sent:",str(len(stanfordParser.docToSentenceTokens[doc_id].keys())))
			print("# ecb sent:", str(len(sorted(self.corpus.docToGlobalSentenceNums[doc_id]))))
			stanTokens = []
			ourTokens = []
			print("STAN:")
			for sent_num in sorted(stanfordParser.docToSentenceTokens[doc_id].keys()):
				for token_num in stanfordParser.docToSentenceTokens[doc_id][sent_num]:
					sToken = stanfordParser.docToSentenceTokens[doc_id][sent_num][token_num]
					if sToken.isRoot == False:
						stanTokens.append(sToken.word)
						print("\t",str(sToken.word))
			print("OURS:")
			for sent_num in sorted(self.corpus.docToGlobalSentenceNums[doc_id]):
				for token in self.corpus.globalSentenceNumToTokens[sent_num]:
					ourTokens.append(token.text)
					print("\t",str(token.text))
			j = 0
			i = 0
			while i < len(ourTokens):
				if j >= len(stanTokens):
					if i == len(ourTokens) - 1 and stanTokens[-1] == "...":
						print("ran out, but it's okay, bc stan ended with ...")
						break
					else:
						print("ran out of stan tokens")
						exit(1)

				# get the words to equal lengths first
				stan = stanTokens[j]
				ours = ourTokens[i]
				if stan == "''":
					stan = "\""
					#print("**ERROR, stan was ''")
				elif stan == "2 1/2":
					#print("**CORRECTING, stan had tab")
					stan = "2 1/2"
				elif stan == "3 1/2":
					#print("**CORRECTING, stan had tab")
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


				while len(ours) != len(stan):
					while len(ours) > len(stan):
						#print("\tstan length is shorter:", str(ours)," vs:",str(stan)," stanlength:",str(len(stan)))
						if j+1 < len(stanTokens):
							stan += stanTokens[j+1]
							if stan == "71/2":
								stan = "7 ½"
							elif stan == "31/2":
								stan = "3½"
							j += 1
							print("\tstan is now:", str(stan))
						else:
							print("\tran out of stanTokens")
							exit(1)

					while len(ours) < len(stan):
						print("\tour length is shorter:",str(ours),"vs:",str(stan),"stanlength:",str(len(stan)))
						if i+1 < len(ourTokens):
							ours += ourTokens[i+1]
							if ours == "31/2":
								ours = "3 1/2"
							elif ours == "21/2":
								print("converted to: 2 1/2")
								ours = "2 1/2"
							elif ours == "31/2-inch":
								ours = "3 1/2-inch"
							elif ours == "3 1/2":
								ours = "3 1/2"
							i += 1
							print("\tours is now:", str(ours))
						else:
							print("\tran out of ourTokens")
							exit(1)	

				if ours != stan:
					print("\tMISMATCH: [",str(ours),"] [",str(stan),"]")
					for i in range(len(ours)):
						print(ours[i],stan[i])
						if ours[i] != stan[i]:
							print("those last ones didnt match!")
					exit(1)
				#else:
				#	print("\t[", str(ours), "] == [", str(stan), "]")

				j += 1
				i += 1

			'''
			if len(stanTokens) != len(ourTokens):
				print("** ERROR, not same length!!",str(len(stanTokens)), " vs ecb's:", str(len(ourTokens)))
				exit(1)
			'''
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