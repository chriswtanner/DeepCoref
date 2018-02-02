from collections import defaultdict

class CheckPredVsGoldHDDCRPMention:
	cdFile = "/Users/christanner/research/DeepCoref/data/predict.CD.semeval.txt"
	wdFile = "/Users/christanner/research/DeepCoref/data/predict.WD.semeval.txt"

	sentToDoc = defaultdict(set)
	dirToDocs = defaultdict(list)
	docToTokens = {}
	f = open(wdFile, 'r')
	tokenString = ""
	tmpTokens = []
	doc = ""
	origSet = set()

	refToDocs = defaultdict(set)
	refDirToWords = defaultdict(set)

	for line in f:
		line = line.rstrip()
		if line.startswith("#begin document"):
			tmpTokens = []
			doc = line[line.find("(") + 1:line.find(")")]
			abridged = doc[0:doc.find("_")] + doc[doc.find("ecb"):doc.find(".xml")]
			dirToDocs[abridged].append(doc)
		elif line == "":
			sentToDoc[tokenString].add(doc)
			tokenString = ""
		elif not line.startswith("#end"): # we have tokens
			tokens = line.split("\t")
			doc, _, tokenNum, text, ref_ = tokens
			dirNum = doc[0:doc.find("_")]

			refs = []
			if ref_.find("|") == -1:
				refs.append(ref_)
			else: # we at most have 1 |
				refs.append(ref_[0:ref_.find("|")])
				refs.append(ref_[ref_.find("|")+1:])

			for ref in refs:
				ref_id = -1
				if ref[0] == "(" and ref[-1] != ")": # i.e. (ref_id
					ref_id = int(ref[1:])
				# represents we are ending a mention
				elif ref[-1] == ")": # i.e., ref_id) or (ref_id)
					ref_id = -1
					if ref[0] != "(": # ref_id)
						ref_id = int(ref[:-1])
					else: # (ref_id)
						ref_id = int(ref[1:-1])
				if ref_id == -1:
					print(ref)
					exit(1)
				refToDocs[ref_id].add(doc)
				refDirToWords[str(ref_id)+str(dirNum)].add(text)
			tmpTokens.append(text)
			tokenString += text + "_"
			origSet.add(line)

		elif line.startswith("#end"):
			docToTokens[doc] = tmpTokens
	f.close()
	for ref in refToDocs:
		print("reF:",str(ref),"docs:",str(refToDocs[ref]))
	exit(1)
	lineCounts = defaultdict(int)
	for i in sentToDoc:
		lineCounts[len(sentToDoc[i])] += 1
	print("the # of unique sentences which are found in how many docs?:",lineCounts)

	dirToTokens = defaultdict(list)
	dirToDocLine = defaultdict(list)
	for d in dirToDocs:
		curTokens = []
		li = sorted(dirToDocs[d])
		for l in li:
			for t in docToTokens[l]:
				curTokens.append(t)
				dirToDocLine[d].append(l)
		dirToTokens[d] = curTokens

	f = open(cdFile, 'r')
	tmpTokens = []
	abridged = ""
	lineNumWithinDir = 0
	lastDoc = ""
	newSet = set()
	for line in f:

		line = line.rstrip()
		if line.startswith("#begin document"):
			tmpTokens = []
			doc = line[line.find("(") + 1:line.find(")")]
			abridged = doc[0:doc.find("_")] + doc[doc.find("ecb"):]
			lineNumWithinDir = 0
			lastDoc = dirToDocLine[abridged][0]

			# we know which dir we're starting now, so we can write them out sequentially
			print("#begin document (" + str(dirToDocLine[abridged][0]) + "); part 000")

		elif line == "":
			print("")

		elif not line.startswith("#end"): # we have tokens
			tokens = line.split("\t")
			doc, _, tokenNum, text, ref_ = tokens
			tmpTokens.append(text)

			newSet.add(line)

			curDoc = dirToDocLine[abridged][lineNumWithinDir]
			
			# checks if we're starting a new doc
			if curDoc != lastDoc:
				print("#end document")
				print("#begin document (" + str(curDoc) + "); part 000")
			print(curDoc + "\t0\t" + str(tokenNum) + "\t" + str(text) + "\t" + str(ref_))
			lineNumWithinDir += 1
			lastDoc = curDoc
		elif line.startswith("#end"):
			if len(dirToTokens[abridged]) != len(tmpTokens):
				print("** LEN",str(abridged))
				print(len(dirToTokens[abridged]),len(tmpTokens))
				for i in range(len(tmpTokens)):
					print(str(i),dirToTokens[abridged][i],"vs",tmpTokens[i])
					if dirToTokens[abridged][i] != tmpTokens[i]:
						print("* ERROR: didnt match up!")
						break
			print("\n#end document")
	f.close()
	print("#orig:",str(len(origSet)))
	print("#new:",str(len(newSet)))
	for l in origSet:
		if l not in newSet:
			print(l)
