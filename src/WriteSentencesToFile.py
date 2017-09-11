import params
from ECBParser import ECBParser
from ECBHelper import ECBHelper

class WriteSentencesToFile:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setWriteSentencesToFileParams()

		# parses corpus
		corpus = ECBParser(args)

		# constructs helper without goldTruthFile info
		helper = ECBHelper(corpus, args)

		helper.writeAllSentencesToFile(args.outputFile)
		'''
		# TODO:
		- ECBHelper should have a function that can readin the stanfordOutput and supplement that info to every token.
		  so each token will be ordered and have info:
		  	- doc-id
		  	- token-id
		  	- text
		  	- head (if we have that info... based on the HDDCRP's file which has heads for every mention)
		    - pos
		    - wordnet (no clue how i'll do this)
		    - dependency parent(s)
		    - dependency child(ren)
		    - lemma
		'''

