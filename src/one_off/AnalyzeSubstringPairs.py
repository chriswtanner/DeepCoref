# i recognize that we miss a lot of the mentions w/ multiple tokens
class AnalyzeSubstringPairs:
	inputFile = "/Users/christanner/research/DeepCoref/results/tmp_allpreds.txt"

	# we want 4 tables:
	# (1) "SSL": single-token mentions (both) - same lemma
	# (2) "SNL": single-token mentions (both) - not same lemma
	# (3) "MSL": multi-token mentions (either) - contains same lemma
	# (4) "MNL": multi-token mentions (either) - does not contain same lemma

	f = open(inputFile, "r")
	for line in f:
		line = line.rstrip()
		
	f.close()