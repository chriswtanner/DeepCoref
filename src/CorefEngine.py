import sys  
import params
import os.path
from ECBParser import *
from ECBHelper import *
from HDDCRPParser import *
from StanParser import *
from CCNN import *
from FFNN import *
from get_coref_metrics import *

# parses the corpus and runs Coref Resoultion on the mentions
class CorefEngine:
	if __name__ == "__main__":

		runFFNN = False

		# handles passed-in args
		args = params.setCorefEngineParams()

		# figures out which mentions (HMentions) HDDCRP thinks exist
		hddcrp_parsed = HDDCRPParser(args.hddcrpFullFile) # loads HDDCRP's pred or gold mentions file

		# parses the real, actual corpus (ECB's XML files)
		corpus = ECBParser(args)
		helper = ECBHelper(args, corpus, hddcrp_parsed)

		if args.SSType != "none":
			helper.createSemanticSpaceSimVectors() # just uses args and corpus
		
		if runFFNN: # deep clustering approach
			
			corefEngine = FFNN(args, corpus, helper, hddcrp_parsed) # instantiates and creates training data
			(testing_pairs, testing_preds, golden_truth) = corefEngine.run()

			#predictedClusters = corefEngine.clusterHPredictions(testing_pairs, testing_preds, sp)
		else:
			# loads stanford's parsed version of our corpus and aligns it w/
			# our representation -- so we can use their features
			stan = StanParser(args, corpus)
			helper.addStanfordAnnotations(stan)

			# trains and tests the pairwise-predictions via Conjoined-CNN
			corefEngine = CCNN(args, corpus, helper, hddcrp_parsed)
			(dev_pairs, dev_preds, testing_pairs, testing_preds) = corefEngine.run()
			#(testing_pairs, testing_preds) = corefEngine.loadPredictions("testall_6647.txt")
		
		# performs agg. clustering on our predicted, testset of HMentions
		stoppingPoints = [0.05,0.11,0.25,0.51,0.75,0.95]
		#stoppingPoints = [0.52,0.54,0.56,0.58,0.601,0.62,0.64,0.66,0.68,0.701,0.72,0.74,0.76,0.78,0.801,0.81]
		#stoppingPoints = [0.15,0.17,0.19,0.21,0.23,0.26,0.28,0.301,0.32,0.34,0.37,0.39,0.401,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.501,0.51,0.52,0.53,0.55,0.57,0.601]
		for sp in stoppingPoints:
			predictedClusters = helper.clusterHPredictions(testing_pairs, testing_preds, sp)
			corefEngine.analyzeResults(testing_pairs, testing_preds, predictedClusters)

			print("* using a agg. threshold cutoff of",str(sp),",we returned # clusters:",str(len(predictedClusters.keys())))
			helper.writeCoNLLFile(predictedClusters, sp)
		print("* done writing all CoNLL file(s); now run ./scorer.pl to evaluate our predictions")






