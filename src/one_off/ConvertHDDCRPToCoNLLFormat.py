
# PURPOSE: convert HDDCRP's output to a format that can be evaluated by CoNLL's scorer.pl script
class ConvertHDDCRPToCoNLLFormat:


	def getLines(fileIn):
		print("in getlines")
		ret = {}
		f = open(fileIn, "r")
		for line in f:
			line = line.rstrip()
			tokens = line.split("\t")
			if line.startswith("#") and "document" in line:
				sentenceNum = 0
			elif len(tokens) == 5:
				uniqID = tokens[0] + "," + str(sentenceNum) + "," + tokens[2] + "," + tokens[3]
				ret[uniqID] = tokens[4]
			else:
				sentenceNum += 1
		f.close()
		return ret

	exit(1)
	pred1 = "/Users/christanner/research/HDPCoref_final/output/predict.WD.semeval.txt"
	pred2 = "/Users/christanner/research/hddcrp_results/predict.WD.semeval.txt"
	gold1 = "/Users/christanner/research/HDPCoref_final/output/gold.WD.semeval.txt"
	gold2 = "/Users/christanner/research/hddcrp_results/gold.WD.semeval.txt"
	outputFile = "/Users/christanner/research/HDPCoref_final/output/prednew.WD.response"

	f = open(pred1, "r")
	g = open(gold1, "r")
	for line in f:
		line = line.rstrip()
		line2 = g.readline().rstrip()
		tokens = line.split("\t")
		tokens2 = line2.split("\t")
		if line.startswith("#") and "document" in line:
			a = 1
		elif len(tokens) == 5:
			pred = tokens[4]
			gold = tokens2[4]

			if 
		else:
			sentenceNum += 1
	f.close()
	g.close()
	exit(1)

	p1map = getLines(pred1)
	g1map = getLines(gold1)
	p2map = getLines(pred2)
	g2map = getLines(gold2)
	numOff = 0
	for k in g2map.keys():
		if p2map[k] != "-" and g2map[k] == "-":
			print("pred:",str(k))
			numOff += 1
	print("numOff:",str(numOff))
