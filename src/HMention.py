# represents a Mention from the HDDCRP pred/gold.semeval format (CoNLL scorer format)
# e.g., a collection of HTokens
class HMention:

	def __init__(self, doc_id, ref_id, tokens, UID, hm_id):
		self.doc_id = doc_id
		self.ref_id = ref_id
		self.tokens = tokens
		self.UID = UID
		self.hm_id = hm_id # used for easily, readably handling/iterating over all HMentions

	def __str__(self):
		text = ""
		for t in self.tokens:
			text += "[" + str(t.text) + "] "
		return("[HMENTION]: " + str(self.doc_id) + "; REF:" + str(self.ref_id) + "; TEXT:" + str(text.rstrip()) + "; UID:" + str(self.UID))