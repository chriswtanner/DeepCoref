# DEFUNCT:
# copied and pasted here from ECBParser, as a better design choice, but
# i haven't worked out the dependencies yet; it needs ECBParser and/or ECBHelper stuff
class StanParser:

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