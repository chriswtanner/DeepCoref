from HToken import *
from HMention import *
from collections import defaultdict
class HDDCRPParser:
	def __init__(self, inputFile):
		self.inputFile = inputFile
		self.parse(inputFile)

	# parses the hddcrp *semeval.txt file (which is in CoNLL-ready format)
	def parse(self, inputFile):

		# global vars
		self.htokens = {}
		self.UIDToHMentions = {}
		self.UIDToToken = {}
		self.hmentions = []
		self.docToHMentions = defaultdict(list)
		self.docToUIDs = defaultdict(list)
		self.hm_idToHMention = {}

		REFToStartIndex = {}
		tokenIndex = 0
		sentenceNum = 0
		hm_id = 0
		f = open(inputFile, "r")
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
			elif line == "":
				sentenceNum += 1
			elif len(tokens) == 5:
				doc, _, tokenNum, text, ref_ = tokens

				# the construction sets a member variable "uid" = doc_id, sentence_id, token_num
				curToken = HToken(doc, sentenceNum, tokenNum, text.lower())
				self.htokens[tokenIndex] = curToken
				self.UIDToToken[curToken.UID] = curToken
				self.docToUIDs[doc].append(curToken.UID)

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))
				for ref in refs:
					if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
						ref_id = int(ref[1:])
						REFToStartIndex[ref_id] = tokenIndex

					# represents we are ending a mention
					elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
						ref_id = -1
						tokens = []
						UID = ""

						# we set ref_if, tokens, UID
						if ref[0] != "(": # ref_id)
							ref_id = int(ref[:-1])
							startIndex = REFToStartIndex[ref_id]

							for i in range(startIndex,tokenIndex+1): # add all tokens, including current one
								tokens.append(self.htokens[i])
								UID += self.htokens[i].UID + ";"

						else: # (ref_id)
							ref_id = int(ref[1:-1])
							tokens.append(curToken)
							UID = curToken.UID + ";"

						curMention = HMention(doc, ref_id, tokens, UID, hm_id)
						self.docToHMentions[doc].append(curMention)
						self.hmentions.append(curMention)
						self.UIDToHMentions[UID] = curMention
						self.hm_idToHMention[hm_id] = curMention
						hm_id += 1
						
				# end of current token line
				tokenIndex += 1 # this always increases whenever we see a token
				
			else:
				print("ERROR: curLine:",str(line))
				exit(1)
		f.close()

		print("* parsed ",str(len(self.hmentions)), "mentions")