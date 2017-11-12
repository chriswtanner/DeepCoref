# PURPOSE: to take as input hte huge pre-trained glove embeddings
# 			and output the subset of word types we care about 
class StripGloveEmbeddings:

	bigGloveFile = "/Users/christanner/research/libraries/glove.840B.300d.txt" # glove.6B/glove.6B.300d.txt
	smallGloveFile = "/Users/christanner/research/DeepCoref/data/wordTokens.txt" #gloveEmbeddings50.txt"

	outputFile = "/Users/christanner/research/DeepCoref/data/wordEmbeddings.840B.300.txt"

	badTokens = {}
	badTokens["cybercriminal"] = ["cyber","criminal"]
	badTokens["alluminum"] = ["aluminum"]
	badTokens["gameball"] = ["game", "ball"]
	badTokens["horrorpop"] = ["horror", "pop"]
	badTokens["drrrrrrrrrrrrrrrrrrrrrrrum"] = ["drum"]
	badTokens["independentscott"] = ["independent", "scott"]
	badTokens["burglarization"] = ["burglary"]
	badTokens["plice"] = ["police"]
	badTokens["moussaouus"] = ["moussaoui"]
	badTokens["horrorpop"] = ["horror", "pop"]
	badTokens["respomsible"] = ["responsible"]
	badTokens["hospitalise"] = ["hospitalize"]
	badTokens["univerasity"] = ["university"]
	badTokens["shadowserver"] = ["shadow", "server"]
	badTokens["shadowservers"] = ["shadow", "servers"]
	badTokens["microserver"] = ["micro", "server"]
	badTokens["microservers"] = ["micro", "servers"]
	badTokens["instagramm"] = ["instagram"]
	badTokens["te-north"] = ["the", "north"]
	badTokens["athttp"] = ["http"]
	badTokens["funnybook"] = ["funny", "book"]
	badTokens["unspectactular"] = ["not", "spectacular"]

	outputMap = {}

	# load steh types we care about
	typesWeCareAbout = set()
	missing = set()
	f = open(smallGloveFile, 'r')
	for line in f:
		tokens = line.rstrip().split(" ")
		#wordType = tokens[0]
		for t in tokens:
			typesWeCareAbout.add(t)
			missing.add(t)
	f.close()
	print("# total we care about:",str(len(typesWeCareAbout)))

	f = open(bigGloveFile, 'r', encoding="utf-8")
	found = set()
	for line in f:
		tokens = line.rstrip().split(" ")
		wordType = tokens[0]
		if wordType in typesWeCareAbout:
			found.add(wordType)
			missing.remove(wordType)
			emb = [float(x) for x in tokens[1:]]
			#wordTypeToEmbedding[wordType] = emb
			outputMap[wordType] = emb
			#self.embeddingLength = len(emb)
	f.close()
	print("missing:",str(len(missing)))
	print("outputMap:",str(len(outputMap.keys())))

	# tries again, but lowercases them (we don't want to lowercase the 1st pass)
	f = open(bigGloveFile, 'r', encoding="utf-8")
	for line in f:
		tokens = line.rstrip().split(" ")
		wordType = tokens[0].lower()
		if wordType in missing:
			found.add(wordType)
			missing.remove(wordType)
			emb = [float(x) for x in tokens[1:]]
			#wordTypeToEmbedding[wordType] = emb
			outputMap[wordType] = emb
			#self.embeddingLength = len(emb)
	f.close()
	print("missing (after lcase'ing:",str(len(missing)))
	print("outputMap:",str(len(outputMap.keys())))
	missingMap = {}
	missingTokens = set()
	for i in missing:
		tokens = []
		if " " in i:
			tokens = i.split(" ")
		else:
			tokens = [i]

		allTokens = []
		for t in tokens:
			if "." in t:
				for _ in t.split("."):
					allTokens.append(_)
			elif "-" in i:
				for _ in t.split("-"):
					allTokens.append(_)
			elif "/" in i:
				for _ in t.split("/"):
					allTokens.append(_)
			else:
				allTokens.append(t)
		for t in allTokens:
			missingTokens.add(t)

		if i in badTokens.keys():
			missingMap[i] = badTokens[i]
		else:
			missingMap[i] = allTokens
		#print(str(i),"=>",str(missingMap[i]))

	# final round
	f = open(bigGloveFile, 'r', encoding="utf-8")
	missingTokenToEmb = {}
	for line in f:
		tokens = line.rstrip().split(" ")
		wordType = tokens[0].lower()
		if wordType in missingTokens:
			emb = [float(x) for x in tokens[1:]]
			missingTokenToEmb[wordType] = emb
	f.close()

	stillMissing = 0
	for m in missing:

		sumEmb = [0]*300
		numMissingFound = 0
		#print("missingMap[m]:",str(missingMap[m]))
		# look at each token
		for t in missingMap[m]:

			# lets add it, if we have it
			if t in missingTokenToEmb.keys():
				sumEmb = [x + y for x,y in zip(sumEmb, missingTokenToEmb[t])]
				numMissingFound += 1
		if numMissingFound > 0:
			avgEmb = []
			for _ in sumEmb:
				avgEmb.append(_ / float(numMissingFound))
			outputMap[m] = avgEmb
		else:
			stillMissing += 1
			outputMap[m] = [0]*300
	print("stillMissing:",str(stillMissing))

	print("outputMap:",str(len(outputMap.keys())))
	fout = open(outputFile, 'w')
	for w in outputMap.keys():
		line = str(w)
		for f in outputMap[w]:
			line += " " + str(f)
		line = line.rstrip()
		fout.write(line + "\n")
	fout.close()