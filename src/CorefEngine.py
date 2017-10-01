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

		a = [0]*50
		b = [1]*50
		c = [2]*50
		a = [x + y for x,y in zip(a,b)]
		a = [x + y for x,y in zip(a,c)]
		#a = [x / 3 for x in a]
		print(a)

		#exit(1)

		# parses corpus
		corpus = ECBParser(args)

		# constructs helper class
		helper = ECBHelper(corpus, args)

		# trains and tests
		corefEngine = SiameseCNN(args, corpus, helper)

