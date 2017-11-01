class StanToken:
	def __init__(self, isRoot, sentenceNum, tokenNum, word, lemma, startIndex, endIndex, pos, ner):
		self.isRoot = isRoot
		self.sentenceNum = sentenceNum
		self.tokenNum = tokenNum
		self.word = word
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
		return("[" + str(self.word) + "]")
