class CheckPredVsGoldHDDCRPMention:
	candFile = "/Users/christanner/research/HDPCoref/input/candidate-mention-table.txt"
	goldFile = "/Users/christanner/research/HDPCoref/input/gold-mention-table.txt"

	candLines = set()

	f = open(candFile, 'r')
	for line in f:
		line = line.rstrip().lower()
		if "#begin document" not in line and "#end document" not in line:
			candLines.add(line)
	f.close()

	f = open(goldFile, 'r')
	numIn = 0
	numOut = 0
	for line in f:
		line = line.rstrip().lower()
		if "#begin document" not in line and "#end document" not in line:
			if line in candLines:
				numIn += 1
			else:
				numOut += 1
				print("out:",str(line))
	f.close()
	print("numIn:", str(numIn))
	print("numOut",str(numOut))
