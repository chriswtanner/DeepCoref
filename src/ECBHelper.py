try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from get_coref_metrics import *
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

	def constructCoNLLClustersFromFile(self, responseFile):
		ret = set()
		f = open(responseFile, 'r')
		f.readline()
		clusterToDMs = defaultdict(set)
		for line in f:
			line = line.rstrip()
			if line == "#end document":
				break
			_, dm, clusterID = line.rstrip().split()
			clusterToDMs[clusterID].add(dm)
		for clusterID in clusterToDMs.keys():
			ret.add(clusterToDMs[clusterID])
		return ret

	def constructCoNLLTestFileWD(self, outputFile):
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

	def constructCoNLLTestFileCD(self, outputFile):
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