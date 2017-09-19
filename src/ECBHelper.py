try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class ECBHelper:

	def __init__(self, corpus, args): # goldTruthFile, goldLegendFile, isVerbose):
		
		# sets passed-in params
		self.corpus = corpus
		self.isVerbose = args.verbose

	# iterates through the corpus, printing 1 sentence per line
	def writeAllSentencesToFile(self, outputFile):
		fout = open(outputFile, 'w')
		lasts = set()
		for sent_num in sorted(self.corpus.globalSentenceNumToTokens.keys()):
			outLine = ""

			if sent_num > 6:
				break

			lastToken = ""
			for t in self.corpus.globalSentenceNumToTokens[sent_num]:
				outLine += t.text + " "
				lastToken = t.text

			lasts.add(lastToken)
			outLine = outLine.rstrip()
			fout.write(outLine + "\n")
		fout.close()

		for i in lasts:
			print str(i)

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