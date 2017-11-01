import params
from ECBParser import ECBParser
from ECBHelper import ECBHelper

class WriteSentencesToFile:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setWriteSentencesToFileParams()

		# parses corpus
		corpus = ECBParser(args)

		for doc in corpus.docToGlobalSentenceNums.keys():
			print("doc:",str(doc),":",str(sorted(corpus.docToGlobalSentenceNums[doc])))
			for sent_num in sorted(corpus.docToGlobalSentenceNums[doc]):
				for t in corpus.globalSentenceNumToTokens[sent_num]:
					print("\t",str(sent_num),": ",str(t))
		
		# constructs helper
		helper = ECBHelper(corpus, args)

		helper.writeAllSentencesToFile(args.writeOutDir)
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

