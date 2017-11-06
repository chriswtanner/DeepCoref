# PURPOSE: to take as input hte huge pre-trained glove embeddings
# 			and output the subset of word types we care about 
class StripGloveEmbeddings:

	bigGloveFile = "/Users/christanner/research/libraries/glove.6B/glove.6B.300d.txt"  # glove.840B.300d.txt" # 
	smallGloveFile = "/Users/christanner/research/DeepCoref/data/lemmaEmbeddings400.txt"

	outputFile = "/Users/christanner/research/DeepCoref/data/gloveEmbeddings.6B.300.txt"

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
	badTokens["horrorpop"] = ["horror", "pop"]

	# load steh types we care about
	typesWeCareAbout = set()
	f = open(smallGloveFile, 'r')
	for line in f:
		tokens = line.rstrip().split(" ")
		wordType = tokens[0]
		typesWeCareAbout.add(wordType)	
	f.close()

	f = open(bigGloveFile, 'r', encoding="utf-8")
	found = set()
	for line in f:
		tokens = line.rstrip().split(" ")
		wordType = tokens[0]
		if wordType in typesWeCareAbout:
			found.add(wordType)
			emb = [float(x) for x in tokens[1:]]
			#wordTypeToEmbedding[wordType] = emb
			#self.embeddingLength = len(emb)
	f.close()
	print("missing:")
	for i in typesWeCareAbout:
		if i not in found:
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
			print(str(i),"=>",str(allTokens))

	print("# total we care about:",str(len(typesWeCareAbout)))
	print("of these, we found:",str(len(found)))


