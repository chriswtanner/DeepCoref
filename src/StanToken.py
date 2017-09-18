class StanToken:
	def __init__(self, sentenceNum, tokenNum, word, lemma, startIndex, endIndex, pos, ner):
		self.sentenceNum = sentenceNum
		self.tokenNum = tokenNum
		self.word = word
		self.lemma = lemma
		self.startIndex = startIndex
		self.endIndex = endIndex
		self.pos = pos
		self.ner = ner

	def __str__(self):
		return("stan token")
