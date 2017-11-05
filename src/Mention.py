class Mention:

	def __init__(self, dirNum, doc_id, m_id, tokens, corpusTokenIndices, text, isPred, mentionType):
		self.dirNum = dirNum
		self.doc_id = doc_id
		self.m_id = m_id
		self.tokens = tokens
		self.corpusTokenIndices = corpusTokenIndices
		self.text = text
		self.isPred = isPred
		self.relativeTokenIndices = []
		self.suffix = doc_id[doc_id.find("ecb"):]
		self.mentionType = mentionType
		self.UID = ""
		for t in self.tokens:
			self.UID += t.UID + ";"

	def setPrevTokenRelativeIndex(self, prevTokenRelativeIndex, prevToken):
		self.prevTokenRelativeIndex = prevTokenRelativeIndex
		self.prevToken = prevToken

	# assumes setPrevTokenRelativeIndex() was already called
	def setMentionRelativeTokenIndices(self):
		#print "orig tokens: " + str(self.orig_tokenIDs)
		for i in range(len(self.orig_tokenIDs)):
			self.relativeTokenIndices.append(self.prevTokenRelativeIndex + i + 1)

	def setRef(self, ref_id):
		self.ref_id = ref_id

	def __str__(self):
		return "MENTION: " + str(self.m_id) + " (dir " + str(self.dirNum) + "; doc: " + str(self.doc_id) + "): text: " + str(self.text) + " corpusIndices: " + str(self.corpusTokenIndices)
		#return "MENTION: REF: " + str(self.ref_id) + "; dirNum:" + str(self.dirNum) + "; doc_id:" + str(self.doc_id) + "; m_id:" + str(self.m_id) + "; token_ids: " + str(self.orig_tokenIDs) + "; prevToken (" + str(self.prevToken) + ") and prevIndex: " + str(self.prevTokenRelativeIndex) + "; text:" + str(self.text) + "; reltokenindices: " + str(self.relativeTokenIndices)