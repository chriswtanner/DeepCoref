# encoding=utf8  
import sys  
import re
import os
import fnmatch
from collections import defaultdict
from Token import Token
from Mention import Mention
from StanToken import StanToken
from StanLink import StanLink

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

#importlib.reload(sys)
#sys.setdefaultencoding('utf8')

class ECBParser:
	def __init__(self, args):
		print("args:", str(args))
		self.ensureAllMentionsPresent = False # this should be true when we're actually using the entire corpus
		# sets global vars
		self.replacements = {}
		self.replacementsSet = set() # for quicker indexing, since we'll do it over every token
		self.endPunctuation = set()
		self.endPunctuation.update(".", "!", "?")
		self.validMentions = set()

		self.loadReplacements(args.replacementsFile)

		self.parseCorpus(args.corpusPath, args.stitchMentions, args.verbose)

	def loadReplacements(self, replacementsFile):

		f = open(replacementsFile, 'r')
		for line in f:
			tokens = line.rstrip().split(" ")
			print("tokens", tokens)
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

	# (1) reads stanford's output, saves it
	# (2) aligns it w/ our sentence tokens
	def parseStanfordOutput(self, stanFile):

		# creates global vars
		self.sentenceTokens = defaultdict(lambda : defaultdict(int))

		tree = ET.ElementTree(file=stanFile)
		root = tree.getroot()

		document = root[0]
		print("doc:", document)
		sentences, corefs = document
		print("sentences:", sentences)
		print("corefs:", corefs)

		for elem in sentences: #tree.iter(tag='sentence'):

			sentenceNum = int(elem.attrib["id"])
			print("FOUND SENT NUM:  ", str(sentenceNum))
			for section in elem:

				# process every token for the given sentence
				if section.tag == "tokens":

					# constructs a ROOT StanToken, which represents the NULL ROOT of the DependencyParse
					rootToken = StanToken(sentenceNum, 0, "ROOT", "ROOT", -1, -1, "-", "-")
					self.sentenceTokens[sentenceNum][0] = rootToken

					for token in section:

						tokenNum = int(token.attrib["id"])
						word = ""
						lemma = ""
						startIndex = -1
						endIndex = -1
						pos = ""
						ner = ""
						
						for item in token:
							if item.tag == "word":
								word = item.text
							elif item.tag == "lemma":
								lemma = item.text
							elif item.tag == "CharacterOffsetBegin":
								startIndex = item.text
							elif item.tag == "CharacterOffsetEnd":
								startIndex = item.text
							elif item.tag == "POS":
								pos = item.text
							elif item.tag == "NER":
								ner = item.text

						for badToken in self.replacementsSet:
							oldWord = word
							if badToken in word:
								word = word.replace(badToken, self.replacements[badToken])
								print("** CHANGED: [", str(oldWord), "] to [", str(word), "]")

						print("word; ", str(word))
						# constructs and saves the StanToken
						stanToken = StanToken(sentenceNum, tokenNum, word, lemma, startIndex, endIndex, pos, ner)
						self.sentenceTokens[sentenceNum][tokenNum] = stanToken

				elif section.tag == "dependencies" and section.attrib["type"] == "basic-dependencies":
					
					# iterates over all dependencies for the given sentence
					for dep in section:
						
						parent, child = dep
						relationship = dep.attrib["type"]

						parentToken = self.sentenceTokens[sentenceNum][int(parent.attrib["idx"])]
						childToken = self.sentenceTokens[sentenceNum][int(child.attrib["idx"])]

						# ensures correctness from Stanford
						if parentToken.word != parent.text:
							for badToken in self.replacementsSet:
								if badToken in parent.text:
									parent.text = parent.text.replace(badToken, self.replacements[badToken])

						if childToken.word != child.text:
							for badToken in self.replacementsSet:
								if badToken in child.text:
									child.text = child.text.replace(badToken, self.replacements[badToken])

											
						if parentToken.word != parent.text or childToken.word != child.text:
							print("STAN's DEPENDENCY TEXT MISMATCHES WITH STAN'S TOKENS")
							print("1", str(parentToken.word))
							print("2", str(parent.text))
							print("3", str(childToken.word))
							print("4", str(child.text))
							exit(1)

						# creates stanford link
						curLink = StanLink(parentToken, childToken, relationship)
						
						parentToken.addChild(curLink)
						childToken.addParent(curLink)

		# iterates through our corpus, trying to align Stanford's tokens
		ourTokens = []
		for sent_num in sorted(self.globalSentenceNumToTokens.keys()):
			print("ours: ", str(sent_num))
			for t in self.globalSentenceNumToTokens[sent_num]:
				ourTokens.append(t.text)
		
		stanTokens = []

		for sent_num in sorted(self.sentenceTokens.keys()):
			print("stan: ", str(sent_num))
			for t in sorted(self.sentenceTokens[sent_num]):
				if t != 0:
					curStan = self.sentenceTokens[sent_num][t].word
					stanTokens.append(curStan)

		#offset = 0
		j = 0 # i + offset
		i = 0
		while i < len(ourTokens):
			if j >= len(stanTokens):
				print("ran out of stan tokens")
				exit(1)

			# get them to equal lengths first
			stan = stanTokens[j]
			ours = ourTokens[i]
			while len(ours) > len(stan):
				print("stan length is less:", str(len(ours)), " vs ", str(len(stan)))
				if j+1 < len(stanTokens):
					stan += stanTokens[j+1]
					j += 1
					print("stan is now:", str(stan))
				else:
					print("ran out of stanTokens")
					exit(1)

			while len(ours) < len(stan):
				print("our length is less")
				if i+1 < len(ourTokens):
					ours += ourTokens[i+1]
					i += 1
					print("ours is now:", str(ours))
				else:
					print("ran out of ourTokens")
					exit(1)	

			if ours != stan:
				print("MISMATCH: [", str(ours), "] [", str(stan), "]")
			else:
				print("[", str(ours), "] == [", str(stan), "]")

			j += 1
			i += 1
		'''
		numMismatched = 0
		for i in range(len(stanTokens) + 15):
			ours = ""
			stans = ""
			if i < len(ourTokens):
				ours = ourTokens[i]
			if i < len(stanTokens):
				stans = stanTokens[i]

			if ours != stans:

				print "****** " + str(i) + " " + str(ours) + " AND " + str(stans)
				numMismatched += 1
			else:
				print str(i) + " " + str(ours) + " AND " + str(stans)
		print "# numMismatched: " + str(numMismatched)
		'''

	def parseCorpus(self, corpusDir, stitchMentions=False, isVerbose=False):
		print("* parseCorpus()")

		# globally sets params
		self.corpusDir = corpusDir
		self.stitchMentions = stitchMentions
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

			'''
			if self.isVerbose:
				print("parsing: ", doc_id, " (file ", str(files.index(f) + 1), " of ", str(len(files)), ")")
				sys.stdout.flush()
			'''

			tmpDocTokens = [] # we will optionally flip these and optionally stitch Mention tokens together
			tmpDocTokenIDsToTokens = {}
			docTokenIDToCorpusIndex = {}

			# opens the xml file and makes needed replacements
			with open (f, 'r') as myfile:
				fileContents=myfile.read().replace('\n',' ')

				for badToken in self.replacementsSet:
					fileContents = fileContents.replace(badToken, self.replacements[badToken])

	        # reads <tokens>
			it = tuple(re.finditer(r"<token t\_id=\"(\d+)\" sentence=\"(\d+)\" number=\"(\d+)\".*?>(.*?)</(.*?)>", fileContents))
			lastSentenceNum = -1

			tokenNum = -1 # numbers every token in each given sentence, starting at 1 (each sentence starts at 1)
			firstToken = True
			lastTokenText = "" # often times, the last token doesn't end in legit punctuation (. ? ! etc)
							   # this causes stanfordCoreNLP to have trouble knowing where to end sentences, so we simply add a terminal '.' when needed now		
			for match in it:
				t_id = match.group(1)
				sentenceNum = int(match.group(2))
				tokenText = match.group(4).lower().rstrip()
				# removes tokens that end in : (e.g., newspaper:) but leaves the atomic ":" alone
				if len(tokenText) > 1 and tokenText[-1] == ":":
					tokenText = tokenText[:-1]

				# TMP
				if sentenceNum > self.docToHighestSentenceNum[doc_id]:
					self.docToHighestSentenceNum[doc_id] = sentenceNum

				if sentenceNum > 0 or "plus" not in doc_id:

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
								endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, ".")
								tmpDocTokens.append(endToken)
							#endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, "<end>")
							#tmpDocTokens.append(endToken)
							globalSentenceNum = globalSentenceNum + 1

						tokenNum = -1
						startToken = Token("-1", sentenceNum, globalSentenceNum, tokenNum, "<start>")
						#tmpDocTokens.append(startToken)
						tokenNum = tokenNum + 1

					# adds token
					curToken = Token(t_id, sentenceNum, globalSentenceNum, tokenNum, tokenText)
					tmpDocTokenIDsToTokens[t_id] = curToken
					firstToken = False
					tmpDocTokens.append(curToken)
					tokenNum = tokenNum + 1
				
				lastSentenceNum = sentenceNum
				lastTokenText = tokenText
				lastToken_id = t_id

			# if sentence ended with an atomic ":", let's change it to a "."
			if lastTokenText == ":":
				lastToken = tmpDocTokenIDsToTokens[lastToken_id]
				lastToken.text = "."
				tmpDocTokenIDsToTokens[lastToken_id] = lastToken
			elif lastTokenText not in self.endPunctuation:
				endToken = Token("-1", lastSentenceNum, globalSentenceNum, tokenNum, ".")
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
					#print "tmpCurrentMentionSpanIDs: " + str(tmpCurrentMentionSpanIDs)
					spanRange = 1 + max(tmpCurrentMentionSpanIDs) - min(tmpCurrentMentionSpanIDs)
					if isVerbose and spanRange != len(tmpCurrentMentionSpanIDs):
						print("*** WARNING: the mention's token range seems to skip over a token id! ", str(tmpCurrentMentionSpanIDs))
					else:

						#print "tmpCurrentMentionSpanIDs:" + str(tmpCurrentMentionSpanIDs)
						# makes a new stitched-together Token
						tokens_stitched_together = []
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]
							tokens_stitched_together.append(cur_token)
						
						stitched_token = Token(-2, -2, -2, -2, "", True, tokens_stitched_together)
						stitchedTokens.append(stitched_token)

						# points the constituent Tokens to its new stitched-together Token
						for token_id in tmpCurrentMentionSpanIDs:
							cur_token = tmpDocTokenIDsToTokens[str(token_id)]

							if cur_token in tokenToStitchedToken.keys():
								if isVerbose:
									print("ERROR: OH NO, the same token id (", str(token_id), ") is used in multiple Mentions!")
								#print tokenToStitchedToken[cur_token]
								#print tmpCurrentMentionSpanIDs
								#exit(1)
							else:
								# print "pointing token id: " + str(token_id) + " to " + str(stitched_token)
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
					if ref_id not in self.dirToREFs[dirNum]:
						self.dirToREFs[dirNum].append(ref_id)

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

		# ensures we have found all of the valid mentions in our corpus
		'''
		if self.ensureAllMentionsPresent: # this should be true when we're actually using the entire corpus
			for m in self.validMentions:
				if m not in self.dmToMention:
					print("* ERROR, our corpus never parsed: ", str(m))
					exit(1)
		'''
		# TEMP
		'''
		responseFile="/Users/christanner/research/DeepCoref/results/test_hddcrp2.response"
		f = open(responseFile, 'r')
		validDMs = set()
		f.readline()
		for line in f:
			line = line.rstrip()
			if line == "#end document":
				break
			_, dm, clusterID = line.rstrip().split()
			validDMs.add(dm)
		print("# valid DMs:",str(len(validDMs)))
		f.close()
		'''


