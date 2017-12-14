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

		runFFNN = True

		#stoppingPoints = [0.401,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.501,0.51,0.52,0.53,0.55,0.57,0.601]
		#stoppingPoints = [0.51]
		stoppingPoints = [0.701,0.72,0.74,0.76,0.78,0.801,0.81,0.83,0.85,0.87,0.88,0.89,0.901,0.91]
		#stoppingPoints = [0.24,0.27,0.301,0.33,0.36,0.39,0.401,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.501,0.51,0.52,0.53,0.55,0.57,0.601,0.62,0.66,0.701,0.72,0.74,0.76,0.78,0.801,0.81]
		#stoppingPoints = [0.12,0.15,0.18,0.21,0.24,0.27,0.301,0.33,0.36,0.39,0.42,0.45,0.47,0.48,0.49,0.501,0.51,0.52,0.53,0.55,0.57,0.601,0.62,0.66,0.701,0.72,0.74,0.76,0.78,0.801,0.81]
		# handles passed-in args
		args = params.setCorefEngineParams()

		# figures out which mentions (HMentions) HDDCRP thinks exist
		hddcrp_parsed = HDDCRPParser(args.hddcrpFullFile) # loads HDDCRP's pred or gold mentions file

		# parses the real, actual corpus (ECB's XML files)
		corpus = ECBParser(args)
		helper = ECBHelper(args, corpus, hddcrp_parsed)

		if args.SSType != "none":
			helper.createSemanticSpaceSimVectors() # just uses args and corpus
		
		# loads stanford's parsed version of our corpus and aligns it w/
		# our representation -- so we can use their features
		stan = StanParser(args, corpus)
		helper.addStanfordAnnotations(stan)

		if runFFNN: # deep clustering approach
			
			# runs CCNN -> FFNN
			if args.useECBTest: # ECB+ test set
				ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed)
				(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
				ffnnEngine = FFNN(args, corpus, helper, hddcrp_parsed, dev_pairs, dev_preds, testing_pairs, testing_preds)
			else: # HDDCRP test set
				ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed)
				(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
				ffnnEngine = FFNN(args, corpus, helper, hddcrp_parsed, dev_pairs, dev_preds, testing_pairs, testing_preds) # reads in a saved prediction file instead
			
			ffnnEngine.train()

			for sp in stoppingPoints:
				(predictedClusters, goldenClusters) = ffnnEngine.cluster(sp)
				print("# goldencluster:",str(len(goldenClusters)))
				print("# predicted:",str(len(predictedClusters)))
				if args.useECBTest: # use corpus' gold test set
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
					print("FFNN F1 sp:",str(sp),"=",str(conll_f1),"OTHERS:",str(muc_f1),str(bcub_f1),str(ceafe_f1))
				else:
					print("FFNN on HDDCRP")
					helper.writeCoNLLFile(predictedClusters, sp)
		else: # AGGLOMERATIVE CLUSTERING

			# trains and tests the pairwise-predictions via Conjoined-CNN
			ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed)
			(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
		
			# performs agg. clustering on our predicted, testset of HMentions
			for sp in stoppingPoints:

				if args.useECBTest: # use corpus' gold test set

					(predictedClusters, goldenClusters) = ccnnEngine.clusterPredictions(dev_pairs, dev_preds, sp)
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
					print("AGG DEV F1 sp:",str(sp),"=",str(conll_f1))
					print("DEV:",str(muc_f1),str(bcub_f1),str(ceafe_f1))
					(predictedClusters, goldenClusters) = ccnnEngine.clusterPredictions(testing_pairs, testing_preds, sp)
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
					print("AGG TEST F1 sp:",str(sp),"=",str(conll_f1))
					print("TEST:",str(muc_f1),str(bcub_f1),str(ceafe_f1))
				else:
					predictedClusters = helper.clusterHPredictions(testing_pairs, testing_preds, sp)
					ccnnEngine.analyzeResults(testing_pairs, testing_preds, predictedClusters)
					print("* using a agg. threshold cutoff of",str(sp),",we returned # clusters:",str(len(predictedClusters.keys())))
					helper.writeCoNLLFile(predictedClusters, sp)
			print("* done writing all CoNLL file(s); now run ./scorer.pl to evaluate our predictions")







