# represents a token from the HDDCRP pred/gold.semeval format (CoNLL scorer format)
class HToken:
	
	# NOTE, tokenNum is wrt current sentence only
	def __init__(self, doc_id, sentenceNum, tokenNum, text):
		self.doc_id = doc_id
		self.sentenceNum = sentenceNum
		self.tokenNum = tokenNum
		self.text = text
		self.UID = str(doc_id) + ";" + str(sentenceNum) + ";" + str(tokenNum)

	def __str__(self):
		return("[HTOKEN] doc:" + str(doc_id) + "; sentenceNum:" + str(sentenceNum) + "; tokenNum:" + str(tokenNum) + "; text:" + str(text))