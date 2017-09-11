import params
from ECBParser import ECBParser
from ECBHelper import ECBHelper

class AlignWithStanford:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setAlignWithStanfordParams()

		corpus = ECBParser(args)

		# constructs helper without goldTruthFile info
		helper = ECBHelper(corpus, None, None, args.verbose)

		helper.readStanfordOutput(args.stanfordFile)