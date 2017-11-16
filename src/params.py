import argparse

# ECBParser requires:
# - corpusPath
# - replacementsFile
# - stitchMentions
# - verbose

# ECBHelper 
def setWriteSentencesToFileParams():
	parser = argparse.ArgumentParser()
	
	# ECBParser params
	parser.add_argument("--corpusPath", help="the corpus dir")
	parser.add_argument("--replacementsFile", help="we replace all instances of these tokens which appear in our corpus -- this is to help standardize the format, useful for creating embeddings and running stanfordCoreNLP")
	parser.add_argument("--stitchMentions", help="treat multi-token mentions as 1 word token", type=str2bool, nargs='?', default="f")
	parser.add_argument("--verbose", help="print a lot of debugging info", type=str2bool, default="f")

	# ECBHelper
	parser.add_argument("--writeOutDir", help="where we will output the files of corpus' tokens", default="")
	parser.add_argument("--posFileOut", help="where we will output the files of corpus' POS tags", default="")

	# StanParser
	parser.add_argument("--stanOutputDir", help="the file that stanfordCoreNLP output'ed")
	return parser.parse_args()

def setCorefEngineParams():
	parser = argparse.ArgumentParser()

	# ECBParser
	parser.add_argument("--corpusPath", help="the corpus dir")
	parser.add_argument("--stoplistFile", help="stoplist file")
	parser.add_argument("--replacementsFile", help="we replace all instances of these tokens which appear in our corpus -- this is to help standardize the format, useful for creating embeddings and running stanfordCoreNLP")
	parser.add_argument("--mentionsFile", help="the subset of mentions we care about (usually just Events)")
	parser.add_argument("--stitchMentions", help="treat multi-token mentions as 1 word token", type=str2bool, nargs='?', default="f")
	parser.add_argument("--verbose", help="print a lot of debugging info", type=str2bool, nargs='?', default="f")

	# StanParser
	parser.add_argument("--stanOutputDir", help="the file that stanfordCoreNLP output'ed")

	# CCNN
	parser.add_argument("--resultsDir", help="the full path to /results")
	parser.add_argument("--hddcrpBaseFile", help="the base filename of hddcrp file to load in (gold or predict)")
	parser.add_argument("--hddcrpFullFile", help="the fullpath of hddcrpBaseFile")
	parser.add_argument("--shuffleTraining", help="determines if our training will be sequentially over dirs or not", type=str2bool, nargs='?')
	parser.add_argument("--numLayers", help="1 or 2 conv sections", type=int)
	parser.add_argument("--poolType", help="avg or max")
	parser.add_argument("--embeddingsBaseFile", help="the basename of the embeddings file")
	parser.add_argument("--embeddingsFile", help="the full path of the embeddings file")
	parser.add_argument("--embeddingsType", help="type or token") # no default, because this could be tricky, so we want to make it deliberate
	parser.add_argument("--numNegPerPos", help="# of neg examples per pos in training (e.g., 1,2,5)", type=int)
	parser.add_argument("--numEpochs", help="type or token", type=int)
	parser.add_argument("--batchSize", help="batchSize", type=int)
	parser.add_argument("--windowSize", help="# of tokens before/after the Mention to use", type=int)
	parser.add_argument("--clusterMethod", help="min, avg, or avgavg")	
	parser.add_argument("--device", help="gpu or cpu")
	parser.add_argument("--dropout", help="initial dropout rate", type=float)
	parser.add_argument("--numFilters", help="num CNN filters", type=int)
	parser.add_argument("--filterMultiplier", help="the \% of how many filters to use at each successive layer", type=float)
	# optionally added features to the CCNN
	parser.add_argument("--featurePOS", help="pos feature: {none,onehot,emb_random,emb_glove}", default="none")
	parser.add_argument("--posType", help="pos feature: {none,sum,avg}", default="none")
	parser.add_argument("--posEmbeddingsFile", help="the POS embeddings file", default="none")
	# note, we used to read in the files for embeddings when we wanted to use 
	# lemma, distinct from the regular word embeddings and dependency embeddings
	# but it makes most sense to just read in 1 set of embeddings (which have all the word types we'd care to use)
	parser.add_argument("--lemmaType", help="pos feature: {none,sum,avg}", default="none")
	parser.add_argument("--dependencyType", help="dependency feature: {none,sum,avg}", default="none")
	parser.add_argument("--charEmbeddingsFile", help="the char embeddings file", default="none")
	parser.add_argument("--charType", help="charEmb feature: {none,concat,avg}", default="none")
	parser.add_argument("--SSType", help="semantic spaceEmb feature: {none,sum,avg}", default="none")
	parser.add_argument("--SSwindowSize", help="semantic space - windowSize: {0,3,5,7}", default="none", type=int)
	parser.add_argument("--SSvectorSize", help="semantic space - vectorSize: {0,100,200,400}", default="none", type=int)
	parser.add_argument("--SSlog", help="use log probabilities or not for semantic space prob", type=str2bool, nargs='?', default="f")
	return parser.parse_args()

# allows for handling boolean params
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')