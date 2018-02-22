import sys
from HToken import *
from HMention import *
from collections import defaultdict
class HDDCRPParser:
	def __init__(self, inputFile):
		print("* loading HDDCRP's mention boundaries file:")
		self.inputFile = inputFile
		self.parse(inputFile)
		print("\t* parsed",str(len(self.hmentions)),"mentions")
		print("\t* created",str(len(self.hm_idToHMention.keys())),"hm_ids!")

		self.loadGold("gold.WD.semeval.txt") # was NS.WD for Choubey comparison # gold.WD.semeval.txt REGULARLY
		#self.makeNewGoldHDDCRP(inputFile, "gold.NS.WD.semeval.txt")
		sys.stdout.flush()

	def makeNewGoldHDDCRP(self, inputFile, outputFile):
		f = open(inputFile, "r")
		fout = open(outputFile, "w")
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				fout.write(line + "\n")
				sentenceNum = 0
			elif line == "":
				fout.write(line + "\n")
				sentenceNum += 1
			elif len(tokens) == 5:
				doc, _, tokenNum, text, ref_ = tokens
				fout.write(str(doc) + "\t" + str(_) + "\t" + str(tokenNum) + "\t" + str(text) + "\t")
				# the construction sets a member variable "uid" = doc_id, sentence_id, token_num
				curToken = HToken(doc, sentenceNum, tokenNum, text.lower())
				self.docToUIDs[doc].append(curToken.UID)

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				isFirst = True
				for ref in refs:
					if ref == "-":
						if not isFirst:
							fout.write("|")
						fout.write("-")
						isFirst = False
						continue
					if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
						ref_id = int(ref[1:])
						if len(self.DOCREFToHM_IDs[(doc,ref_id)]) > 1:
							if not isFirst:
								fout.write("|")
							fout.write("(" + str(ref_id))
						else:
							fout.write("-")

					# represents we are ending a mention
					elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
						ref_id = -1
						if ref[0] != "(": # ref_id)
							ref_id = int(ref[:-1])
							if ref_id == "-" or len(self.DOCREFToHM_IDs[(doc,ref_id)]) > 1:
								if not isFirst:
									fout.write("|")
								fout.write(str(ref_id) + ")")
							else:
								fout.write("-")

						else: # (ref_id)
							ref_id = int(ref[1:-1])
							if len(self.DOCREFToHM_IDs[(doc,ref_id)]) > 1:
								if not isFirst:
									fout.write("|")
								fout.write("(" + str(ref_id) + ")")
							else:
								fout.write("-")
					isFirst = False
				fout.write("\n")
				
			else:
				print("ERROR: curLine:",str(line))
				exit(1)


	# parses the hddcrp *semeval.txt file (which is in CoNLL-ready format)
	def loadGold(self, goldFile):
		sys.stdout.flush()

		# global vars
		self.gold_htokens = {}
		self.gold_MUIDToHMentions = {} # only used for comparing against HDDCRP's gold mention boundaries
		self.gold_UIDToToken = {}
		self.gold_hmentions = []
		self.gold_docToHMentions = defaultdict(list)
		self.gold_docToUIDs = defaultdict(list)
		self.gold_hm_idToHMention = {}

		REFToStartTuple = defaultdict(list)
		tokenIndex = 0
		sentenceNum = 0
		hm_id = 0

		f = open(goldFile, "r")
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
				self.gold_htokens[tokenIndex] = curToken
				self.gold_UIDToToken[curToken.UID] = curToken
				self.gold_docToUIDs[doc].append(curToken.UID)

				# TMP: only used for analyzeResults() in CCNN (to see the original sentences)
				self.docSentences[doc][sentenceNum].append(text.lower())

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				isFirst = True
				for ref in refs:
					if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
						ref_id = int(ref[1:])

						# only store them if it's truly the un-finished start of a Mention,
						# which will later be closed.  otherwise, we don't need to store it, as
						# it'll be a () on the same line
						REFToStartTuple[ref_id].append((tokenIndex,isFirst))

					# represents we are ending a mention
					elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
						ref_id = -1
						tokens = []
						MUID = ""
						endTuple = (tokenIndex,isFirst)
						startTuple = ()
						# we set ref_if, tokens, UID
						if ref[0] != "(": # ref_id)
							ref_id = int(ref[:-1])
							startTuple = REFToStartTuple[ref_id].pop()

							for i in range(startTuple[0],tokenIndex+1): # add all tokens, including current one
								tokens.append(self.gold_htokens[i])
								MUID += self.gold_htokens[i].UID + ";"

						else: # (ref_id)
							ref_id = int(ref[1:-1])
							startTuple = (tokenIndex,isFirst)
							tokens.append(curToken)
							MUID = curToken.UID + ";"

						curMention = HMention(doc, ref_id, tokens, MUID, hm_id, startTuple, endTuple)
						self.gold_docToHMentions[doc].append(curMention)
						self.gold_hmentions.append(curMention)
						self.gold_MUIDToHMentions[MUID] = curMention
						self.gold_hm_idToHMention[hm_id] = curMention
						hm_id += 1

					isFirst = False
				# end of current token line
				tokenIndex += 1 # this always increases whenever we see a token
				
			else:
				print("ERROR: curLine:",str(line))
				exit(1)
		f.close()
		hms = set()
		for doc_id in self.gold_docToHMentions.keys():
			for hm in self.gold_docToHMentions[doc_id]:
				hms.add(hm)
		print("\t# hms by end of parsing, based on a per doc basis:", str(len(hms)))

	# parses the hddcrp *semeval.txt file (which is in CoNLL-ready format)
	def parse(self, inputFile):
		sys.stdout.flush()

		# global vars
		self.htokens = {}
		self.MUIDToHMentions = {} # only used for comparing against HDDCRP's gold mention boundaries
		self.UIDToToken = {}
		self.hmentions = []
		self.docToHMentions = defaultdict(list)
		self.docToUIDs = defaultdict(list)
		self.hm_idToHMention = {}

		self.dirToDocs = defaultdict(set)

		REFToStartTuple = defaultdict(list)
		tokenIndex = 0
		sentenceNum = 0
		hm_id = 0

		# we produce a new HDDCRP-mentions files which is only temporarily used
		# to test against Choubey's system.  in this new file, we remove all 
		# gold mentions which only co-ref with itself (singleton).  for this, we need this var:
		self.DOCREFToHM_IDs = defaultdict(set)

		# TMP: only used for analyzeResults() in CCNN (to see the original sentences)
		self.docSentences = defaultdict(lambda : defaultdict(list))

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

				dir_num = doc[0:doc.find("_")]

				# the construction sets a member variable "uid" = doc_id, sentence_id, token_num
				curToken = HToken(doc, sentenceNum, tokenNum, text.lower())
				self.htokens[tokenIndex] = curToken
				self.UIDToToken[curToken.UID] = curToken
				self.docToUIDs[doc].append(curToken.UID)

				# TMP: only used for analyzeResults() in CCNN (to see the original sentences)
				self.docSentences[doc][sentenceNum].append(text.lower())

				refs = []
				if ref_.find("|") == -1:
					refs.append(ref_)
				else: # we at most have 1 |
					refs.append(ref_[0:ref_.find("|")])
					refs.append(ref_[ref_.find("|")+1:])
					#print("***** FOUND 2:",str(line))

				isFirst = True
				for ref in refs:
					if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
						ref_id = int(ref[1:])

						# only store them if it's truly the un-finished start of a Mention,
						# which will later be closed.  otherwise, we don't need to store it, as
						# it'll be a () on the same line
						REFToStartTuple[ref_id].append((tokenIndex,isFirst))

					# represents we are ending a mention
					elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
						ref_id = -1
						tokens = []
						MUID = ""
						endTuple = (tokenIndex,isFirst)
						startTuple = ()
						# we set ref_if, tokens, UID
						if ref[0] != "(": # ref_id)
							ref_id = int(ref[:-1])
							startTuple = REFToStartTuple[ref_id].pop()

							for i in range(startTuple[0],tokenIndex+1): # add all tokens, including current one
								tokens.append(self.htokens[i])
								MUID += self.htokens[i].UID + ";"

						else: # (ref_id)
							ref_id = int(ref[1:-1])
							startTuple = (tokenIndex,isFirst)
							tokens.append(curToken)
							MUID = curToken.UID + ";"

						curMention = HMention(doc, ref_id, tokens, MUID, hm_id, startTuple, endTuple)
						self.docToHMentions[doc].append(curMention)
						self.hmentions.append(curMention)
						self.MUIDToHMentions[MUID] = curMention
						self.hm_idToHMention[hm_id] = curMention
						self.DOCREFToHM_IDs[(doc,ref_id)].add(hm_id)
						self.dirToDocs[dir_num].add(doc)
						hm_id += 1

					isFirst = False
				# end of current token line
				tokenIndex += 1 # this always increases whenever we see a token
				
			else:
				print("ERROR: curLine:",str(line))
				exit(1)
		f.close()
		hms = set()
		for doc_id in self.docToHMentions.keys():
			for hm in self.docToHMentions[doc_id]:
				hms.add(hm)
		print("\t# hms by end of parsing, based on a per doc basis:", str(len(hms)))

		# now let's make a new GOLD HDDCRP file, where we remove singletons
