# encoding=utf8  
import sys  

import params
from ECBParser import ECBParser
from ECBHelper import ECBHelper

reload(sys)  
sys.setdefaultencoding('utf8')

class AlignWithStanford:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setAlignWithStanfordParams()

		# parses corpus
		corpus = ECBParser(args)

		corpus.parseStanfordOutput(args.stanfordFile)

		# constructs helper without goldTruthFile info
		#helper = ECBHelper(corpus, args)
		#helper.readStanfordOutput(args.stanfordFile)