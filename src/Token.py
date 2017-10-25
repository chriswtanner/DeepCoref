class Token:
	def __init__(self, tokenID, sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, text, stitchedTogether=False, tokens=[]):
		self.stitchedTogether = stitchedTogether
		self.tokens = tokens
		self.tokenNum = tokenNum # NOTE, this is relative to the given sentence <start> = 1  the = 2 (1-based)
		self.hSentenceNum = hSentenceNum
		self.hTokenNum = hTokenNum
		self.doc_id = doc_id
		self.UID = str(self.doc_id) + ";" + str(self.hSentenceNum) + ";" + str(self.hTokenNum)

		if self.stitchedTogether == True:
			self.sentenceNum = self.tokens[0].sentenceNum
			self.globalSentenceNum = self.tokens[0].globalSentenceNum
			text = ""
			tokenID = ""
			for t in tokens:
				text = text + t.text + "_"
				tokenID = tokenID + t.tokenID + ","
			text = text[:-1]
			tokenID = tokenID[:-1]
			self.text = text
			self.tokenID = tokenID # given in the XML
		else:
			self.tokenID = tokenID # given in the XML
			self.sentenceNum = sentenceNum
			self.globalSentenceNum = globalSentenceNum
			self.text = text

	def __str__(self):
		return("TOKEN: ID:" + str(self.tokenID) + "; TOKEN#: " + str(self.tokenNum) + "; SENTENCE#:" + str(self.sentenceNum) + " globalSentenceNum: " + str(self.globalSentenceNum) + "; TEXT:" + str(self.text))
