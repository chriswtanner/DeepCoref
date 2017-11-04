class StanToken:
	def __init__(self, isRoot, sentenceNum, tokenNum, text, lemma, startIndex, endIndex, pos, ner):
		self.isRoot = isRoot
		self.sentenceNum = sentenceNum
		self.tokenNum = tokenNum
		self.text = text
		self.lemma = lemma
		self.startIndex = startIndex
		self.endIndex = endIndex
		self.pos = pos
		self.ner = ner

		# StanLinks
		self.parentLinks = []
		self.childLinks = []

	def addParent(self, parentLink):
		self.parentLinks.append(parentLink)

	def addChild(self, childLink):
		self.childLinks.append(childLink)

	def __str__(self):
		return("STAN TEXT: [" + str(self.text) + "]" + "; LEMMA:" + str(self.lemma) + "; POS:" + str(self.pos) + "; NER:" + str(self.ner))
