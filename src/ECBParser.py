# encoding=utf8  
import sys  
import re
import os
import fnmatch
import codecs
from collections import defaultdict
from Token import Token
from Mention import Mention
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

class ECBParser:
	def __init__(self, args):
		print("args:", str(args))
		self.args = args
		self.ensureAllMentionsPresent = False # this should be true when we're actually using the entire corpus
		self.padCorpus = False

		# sets global vars
		self.replacements = {}
		self.replacementsSet = set() # for quicker indexing, since we'll do it over every token
		self.endPunctuation = set()
		self.endPunctuation.update(".", "!", "?")
		self.validMentions = set()

		# invokes functions
		self.loadReplacements(args.replacementsFile)
		self.parseCorpus(args.corpusPath, args.stitchMentions, args.verbose)

	def loadReplacements(self, replacementsFile):
		f = open(replacementsFile, 'r', encoding="utf-8")
		for line in f:
			tokens = line.rstrip().split(" ")
			# print("tokens", tokens)
			self.replacements[tokens[0]] = tokens[1]
			self.replacementsSet.add(tokens[0])
		f.close()

	def getGlobalTypeID(self, wordType):
		if wordType in self.typeToGlobalID.keys():
			return self.typeToGlobalID[wordType]
		else:
			newID = len(self.typeToGlobalID.keys())
			self.typeToGlobalID[wordType] = newID
			self.globalIDsToType[newID] = wordType
			return newID

	def parseCorpus(self, corpusDir, stitchMentions=False, isVerbose=False):
		print("* parseCorpus()")

		# globally sets params
		self.corpusDir = corpusDir
		self.stitchMentions = stitchMentions # NOTE: this may not work anymore, as i added a lot of things after last testing this
		self.isVerbose = isVerbose

		# data structures we'll fill now
		self.numCorpusTokens = 0 # used so we don't have to do len(corpusTokens) all the time
		self.corpusTokens = []
		self.corpusTokensToCorpusIndex = {}

		self.mentions = []
		self.dmToMention = {} # (doc_id,m_id) -> Mention
		self.dmToREF = {}
		self.refToDMs = defaultdict(list)
		self.dirToREFs = defaultdict(list)

		self.dirToDocs = defaultdict(list)
		self.docToGlobalSentenceNums = defaultdict(set)
		self.docToREFs = defaultdict(list)
		self.docREFsToDMs = defaultdict(list) # key: (doc_id,ref_id) -> [(doc_id1,m_id1), ... (doc_id3,m_id3)]
		self.docToDMs = defaultdict(list)
		self.docToUIDs = defaultdict(list)
		self.UIDToMentions = {} #
		self.UIDToToken = {}

		# same tokens as corpusTokens, just made into lists according
		# to each doc.  (1 doc = 1 list of tokens); used for printing corpus to .txt file
		self.docTokens = []

		self.typeToGlobalID = {}
		self.globalIDsToType = {}
		self.corpusTypeIDs = []

		self.docToHighestSentenceNum = defaultdict(int) # TMP -- just for creating
			# an aligned goldTruth from HDDCRP
		self.globalSentenceNumToTokens = defaultdict(list) # added so that we can
			# easily parse the original sentence which contains each Mention

		files = []
		for root, dirnames, filenames in os.walk(corpusDir):
			for filename in fnmatch.filter(filenames, '*.xml'):
				files.append(os.path.join(root, filename))

		globalSentenceNum = 0

		for f in files:
			doc_id = f[f.rfind("/") + 1:]
			dir_num = int(doc_id.split("_")[0])
			if dir_num != self.args.tmpDir:
				continue
			self.dirToDocs[dir_num].append(doc_id)

			tmpDocTokens = [] # we will optionally flip these and optionally stitch Mention tokens together
			tmpDocTokenIDsToTokens = {}
			docTokenIDToCorpusIndex = {}

			# opens the xml file and makes needed replacements

			with open (f, 'r', encoding="utf-8") as myfile:
				fileContents=myfile.read().replace('\n',' ')

				for badToken in self.replacementsSet:
					fileContents = fileContents.replace(badToken, self.replacements[badToken])

	        # reads <tokens>
			it = tuple(re.finditer(r"<token t\_id=\"(\d+)\" sentence=\"(\d+)\" number=\"(\d+)\".*?>(.*?)</(.*?)>", fileContents))
			lastSentenceNum = -1

			tokenNum = -1 # numbers every token in each given sentence, starting at 1 (each sentence starts at 1)
			if self.padCorpus == False:
				tokenNum = 0
			firstToken = True
			lastTokenText = "" # often times, the last token doesn't end in legit punctuation (. ? ! etc)
							   # this causes stanfordCoreNLP to have trouble knowing where to end sentences, so we simply add a terminal '.' when needed now		
			for match in it:
				t_id = match.group(1)
				sentenceNum = int(match.group(2))
				hTokenNum = int(match.group(3)) # only used for matching w/ HDDCRP's files
				tokenText = match.group(4).lower().rstrip()
				# removes tokens that end in : (e.g., newspaper:) but leaves the atomic ":" alone
				if len(tokenText) > 1 and tokenText[-1] == ":":
					tokenText = tokenText[:-1]

				if tokenText == "''":
					tokenText = "\""

				# TMP
				if sentenceNum > self.docToHighestSentenceNum[doc_id]:
					self.docToHighestSentenceNum[doc_id] = sentenceNum

				if sentenceNum > 0 or "plus" not in doc_id:

					hSentenceNum = sentenceNum
					if "plus" in doc_id:
						hSentenceNum = sentenceNum - 1

					# we are starting a new sentence
					if sentenceNum != lastSentenceNum:
						
						# we are possibly ending the prev sentence
						if not firstToken:

							# if sentence ended with an atomic ":", let's change it to a "."
							if lastTokenText == ":":
								lastToken = tmpDocTokenIDsToTokens[lastToken_id]
								lastToken.text = "."
								tmpDocTokenIDsToTokens[lastToken_id] = lastToken
							elif lastTokenText not in self.endPunctuation:
								endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, ".")
								tmpDocTokens.append(endToken)

							if self.padCorpus:
								endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, "<end>")
								tmpDocTokens.append(endToken)


							globalSentenceNum = globalSentenceNum + 1

						tokenNum = -1
						if self.padCorpus:
							startToken = Token("-1", sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, "<start>")
							tmpDocTokens.append(startToken)
							tokenNum = tokenNum + 1
						else:
							tokenNum = 0

					# adds token
					curToken = Token(t_id, sentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, tokenText)
					self.UIDToToken[curToken.UID] = curToken
					self.docToUIDs[doc_id].append(curToken.UID)
					tmpDocTokenIDsToTokens[t_id] = curToken
					firstToken = False
					tmpDocTokens.append(curToken)
					tokenNum = tokenNum + 1
					self.docToGlobalSentenceNums[doc_id].add(globalSentenceNum)
				lastSentenceNum = sentenceNum
				lastTokenText = tokenText
				lastToken_id = t_id

			# if sentence ended with an atomic ":", let's change it to a "."
			if lastTokenText == ":":
				lastToken = tmpDocTokenIDsToTokens[lastToken_id]
				lastToken.text = "."
				tmpDocTokenIDsToTokens[lastToken_id] = lastToken
			elif lastTokenText not in self.endPunctuation:
				endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, -1, -1, ".")
				tmpDocTokens.append(endToken)
			if self.padCorpus:
				endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, doc_id, hSentenceNum, hTokenNum, "<end>")
				tmpDocTokens.append(endToken)

			globalSentenceNum = globalSentenceNum + 1

			'''
			if isVerbose:
				print("\t", str(len(tmpDocTokens)), " doc tokens")
				print("\t# unique token ids: ", str(len(tmpDocTokenIDsToTokens)))
			'''

			tokenToStitchedToken = {} # will assign each Token to its stitchedToken (e.g., 3 -> [3,4], 4 -> [3,4])
			stitchedTokens = []

			# reads <markables> 1st time
			regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
			markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
			it = tuple(re.finditer(regex, markables))
			for match in it:
				# gets the token IDs
				regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(3)))
				tmpCurrentMentionSpanIDs = []
				text = []
				hasAllTokens = True
				for match2 in it2:
					tokenID = match2.group(1)
					tmpCurrentMentionSpanIDs.append(int(tokenID))
					if tokenID not in tmpDocTokenIDsToTokens.keys():
						hasAllTokens = False

				# we should only have incomplete Mentions for our hand-curated, sample corpus, 
				# for we do not want to have all mentions, so we curtail the sentences of tokens
				if hasAllTokens and self.stitchMentions and len(tmpCurrentMentionSpanIDs) > 1:
					tmpCurrentMentionSpanIDs.sort()
					spanRange = 1 + max(tmpCurrentMentionSpanIDs) - min(tmpCurrentMentionSpanIDs)
					if isVerbose and spanRange != len(tmpCurrentMentionSpanIDs):
						print("*** WARNING: the mention's token range seems to skip over a token id! ", str(tmpCurrentMentionSpanIDs))
					else:
						# makes a new stitched-together Token
						tokens_stitched_together = []
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]
							tokens_stitched_together.append(cur_token)
						
						stitched_token = Token(-2, -2, -2, -2, doc_id, -2, -2, "", True, tokens_stitched_together)
						stitchedTokens.append(stitched_token)
						self.UIDToToken[stitched_token.UID] = stitched_token

						# points the constituent Tokens to its new stitched-together Token
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]

							if cur_token in tokenToStitchedToken.keys():
								if isVerbose:
									print("ERROR: OH NO, the same token id (", str(token_id), ") is used in multiple Mentions!")
							else:
								tokenToStitchedToken[cur_token] = stitched_token

			if isVerbose and len(stitchedTokens) > 0:
				print("# stitched tokens: ", str(len(stitchedTokens)))
				for st in stitchedTokens:
					print(st)
				print("-----------")

			# ADDS to the corpusTokens in the correct, optionally reversed, optionally stitched, manner
			# puts stitched tokens in the right positions
			# appends to the docTokens[] (each entry is a list of the current doc's tokens);
			# only used for writing out the .txt files corpus
			curDocTokens = []
			if self.stitchMentions:
				completedTokens = set()
				for t in tmpDocTokens:
					if t not in completedTokens:
						if t in tokenToStitchedToken.keys():

							stitched_token = tokenToStitchedToken[t]
							self.corpusTokens.append(stitched_token)
							curDocTokens.append(stitched_token)

							#print "constitutents: " + str(stitched_token.tokens)
							for const_token in stitched_token.tokens:
								completedTokens.add(const_token)
								# assigns all constituent Tokens to have the same index, because they do (w/ the stitchedToken)
								self.corpusTokensToCorpusIndex[const_token] = self.numCorpusTokens

							self.numCorpusTokens = self.numCorpusTokens + 1
						else:
							self.corpusTokens.append(t)
							curDocTokens.append(t)

							self.corpusTokensToCorpusIndex[t] = self.numCorpusTokens
							self.numCorpusTokens = self.numCorpusTokens + 1
			else:
				for t in tmpDocTokens:
					self.corpusTokens.append(t)
					curDocTokens.append(t)
					self.corpusTokensToCorpusIndex[t] = self.numCorpusTokens
					self.numCorpusTokens = self.numCorpusTokens + 1

			self.docTokens.append(curDocTokens)

			# reads <markables> 2nd time
			regex = r"<([\w]+) m_id=\"(\d+)?\".*?>(.*?)?</.*?>"
			markables = fileContents[fileContents.find("<Markables>")+11:fileContents.find("</Markables>")]
			it = tuple(re.finditer(regex, markables))
			for match in it:
				isPred = False
				entityType = match.group(1)
				if "ACTION" in entityType:
					isPred = True
				m_id = int(match.group(2))

				# gets the token IDs
				regex2 = r"<token_anchor t_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(3)))
				tmpMentionCorpusIndices = []
				tmpTokens = [] # can remove after testing if our corpus matches HDDCRP's
				text = []
				hasAllTokens = True
				for match2 in it2:
					tokenID = match2.group(1)

					if tokenID in tmpDocTokenIDsToTokens.keys():
						
						cur_token = tmpDocTokenIDsToTokens[tokenID]
						tmpTokens.append(cur_token)
						text.append(cur_token.text)

						# gets corpusIndex (we ensure we don't add the same index twice, which would happen for stitched tokens)
						tmpCorpusIndex = self.corpusTokensToCorpusIndex[cur_token]
						if tmpCorpusIndex not in tmpMentionCorpusIndices: 
							tmpMentionCorpusIndices.append(tmpCorpusIndex)
					else:
						hasAllTokens = False

				# we should only have incomplete Mentions for our hand-curated, sample corpus, 
				# for we do not want to have all mentions, so we curtail the sentences of tokens
				if hasAllTokens:

					tmpMentionCorpusIndices.sort() # regardless of if we reverse the corpus or not, these indices should be in ascending order

					curMention = Mention(dir_num, doc_id, m_id, tmpTokens, tmpMentionCorpusIndices, text, isPred, entityType)
					# we only save the Mentions that are in self.validMentions,
					# that way, we can always iterate over self.mentions (since we care about them all)
					if isPred:
					#if (doc_id,m_id) in self.validMentions:
						self.validMentions.add(curMention)
						self.mentions.append(curMention)
						self.dmToMention[(doc_id,m_id)] = curMention

			# reads <relations>
			relations = fileContents[fileContents.find("<Relations>"):fileContents.find("</Relations>")]
			regex = r"<CROSS_DOC_COREF.*?note=\"(.+?)\".*?>(.*?)?</.*?>"
			it = tuple(re.finditer(regex, relations))
			for match in it:
				ref_id = match.group(1)

				regex2 = r"<source m_id=\"(\d+)\".*?/>"
				it2 = tuple(re.finditer(regex2, match.group(2)))

				# only keep track of REFs for which we have found Mentions
				for match2 in it2:
					m_id = int(match2.group(1))
					if (doc_id,m_id) not in self.dmToMention.keys():
						#print("*** MISSING MENTION!")
						continue
					self.dmToREF[(doc_id,m_id)] = ref_id
					self.refToDMs[ref_id].append((doc_id,m_id))
					dirNum = int(doc_id[0:doc_id.find("_")])

					# stores the REF for the current doc_id
					if ref_id not in self.docToREFs[doc_id]:
						self.docToREFs[doc_id].append(ref_id)

					# stores the DM for the current (doc_id,ref_id) pair
					# this is so that we can easily make training/dev/test pairs
					if (doc_id,m_id) not in self.docREFsToDMs[(doc_id,ref_id)]:
						self.docREFsToDMs[(doc_id,ref_id)].append((doc_id,m_id))

					# stores the REF for the current DIRECTORY
					if ref_id not in self.dirToREFs[dirNum]:
						self.dirToREFs[dirNum].append(ref_id)

					if (doc_id,m_id) not in self.docToDMs[doc_id]:
						self.docToDMs[doc_id].append((doc_id,m_id))
			#if globalSentenceNum > 2:
			#	print "globalSentenceNum: " + str(globalSentenceNum)
			#	break
		# (1) sets the corpus type-based variables AND
		# (2) sets the globalSentenceTokens so that we can parse the original sentences
		# (3) sets the startIndex and endIndex based on what we'd output
		
		for t in self.corpusTokens:
			# (1)
			g_id = self.getGlobalTypeID(t.text)
			self.corpusTypeIDs.append(g_id)
			
			# (2)
			self.globalSentenceNumToTokens[int(t.globalSentenceNum)].append(t)

		for m in self.mentions:
			self.UIDToMentions[m.UID] = m
		# ensures we have found all of the valid mentions in our corpus
		'''
		if self.ensureAllMentionsPresent: # this should be true when we're actually using the entire corpus
			for m in self.validMentions:
				if m not in self.dmToMention:
					print("* ERROR, our corpus never parsed: ", str(m))
					exit(1)
		'''



