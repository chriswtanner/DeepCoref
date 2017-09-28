import sys  

import params
from ECBParser import *
from ECBHelper import *
from SiameseCNN import *
# parses the corpus and runs Coref Resoultion on the mentions
class CorefEngine:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setCorefEngineParams()

		# parses corpus
		corpus = ECBParser(args)

		# constructs helper class
		helper = ECBHelper(corpus, args)

		# trains and tests
		corefEngine = SiameseCNN(args, corpus, helper)

