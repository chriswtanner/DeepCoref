import sys  
import params
import os.path
from ECBParser import *
from ECBHelper import *
from HDDCRPParser import *
from StanParser import *
from CCNN import *
from FFNNWD import *
from FFNNCD import *
from get_coref_metrics import *
from collections import defaultdict

# Coreference Resolution System for Events (uses ECB+ corpus)
class CorefEngine:
	if __name__ == "__main__":
		isWDModel = False
		runFFNN = True # if False, we will use Agglomerative Cluster
		stoppingPoints = [0.501] #[0.15,0.201,0.25,0.275,0.301,0.325,0.35,0.375,0.401,0.45,0.501,0.55,0.601]
		# handles passed-in args
		args = params.setCorefEngineParams()

		# figures out which mentions (HMentions) HDDCRP thinks exist
		hddcrp_parsed = HDDCRPParser(args.hddcrpFullFile) # loads HDDCRP's pred or gold mentions file

		# parses the real, actual corpus (ECB's XML files)
		corpus = ECBParser(args)
		helper = ECBHelper(args, corpus, hddcrp_parsed, runFFNN)
		
		# loads stanford's parsed version of our corpus and aligns it w/
		# our representation -- so we can use their features
		stan = StanParser(args, corpus)
		helper.addStanfordAnnotations(stan)

		if runFFNN: # deep clustering approach
			print("* FFNN MODE")

			# runs CCNN -> FFNN
			if args.useECBTest: # ECB+ test set
				ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed, isWDModel) # True creates a WD-CCNN model
				(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
				ffnnEngine = FFNNCD(args, corpus, helper, hddcrp_parsed, dev_pairs, dev_preds, testing_pairs, testing_preds)
			else: # HDDCRP test set
				ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed, isWDModel) # True creates a WD-CCNN model
				(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()
				ffnnEngine = FFNNCD(args, corpus, helper, hddcrp_parsed, dev_pairs, dev_preds, testing_pairs, testing_preds) # reads in a saved prediction file instead
			
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
			print("* AGGLOMERATIVE CLUSTERING MODE")
			# trains and tests the pairwise-predictions via Conjoined-CNN

			ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed, isWDModel) # creates CD-CCNN model
			(dev_pairs, dev_preds, testing_pairs, testing_preds) = ccnnEngine.run()

			if args.useECBTest: # use corpus' gold test set
				bestDevSP = -1
				bestDevF1 = -1
				bestTestSP = -1
				bestTestF1 = -1
				testSPToF1 = {}
				for sp in stoppingPoints:
					
					# DEV
					(predictedClusters, goldenClusters) = ccnnEngine.clusterPredictions(dev_pairs, dev_preds, sp)
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
					print("AGG DEV F1 sp:",str(sp),"=",str(conll_f1),"MUC:",str(muc_f1),"BCUB:",str(bcub_f1),"CEAF:",str(ceafe_f1))
					if conll_f1 > bestDevF1:
						bestDevF1 = conll_f1
						bestDevSP = sp

					# TEST
					(predictedClusters, goldenClusters) = ccnnEngine.clusterPredictions(testing_pairs, testing_preds, sp)
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(goldenClusters, predictedClusters)
					testSPToF1[sp] = conll_f1
					if conll_f1 > bestTestF1:
						bestTestF1 = conll_f1
						bestTestSP = sp
					print("AGG TEST F1 sp:",str(sp),"=",str(conll_f1),"MUC:",str(muc_f1),"BCUB:",str(bcub_f1),"CEAF:",str(ceafe_f1))

				print("[FINAL RESULTS]: BEST DEV SP:",str(bestDevSP),"yielded F1:",str(bestDevF1))
				print("[FINAL RESULTS]: ON TEST, this yielded:",str(testSPToF1[bestDevSP]))
				print("[FINAL RESULTS]: MAX TEST SP:",str(bestTestSP),"yielded F1:",str(bestTestF1))
			else: # test on HDDCRP's predicted mention boundaries
				for sp in stoppingPoints:
					predictedClusters = helper.clusterHPredictions(testing_pairs, testing_preds, sp, ccnnEngine.isWDModel)
					ccnnEngine.analyzeResults(testing_pairs, testing_preds, predictedClusters)
					print("* using a agg. threshold cutoff of",str(sp),",we returned # clusters:",str(len(predictedClusters.keys())))
					helper.writeCoNLLFile(predictedClusters, sp)
					helper.convertWDFileToCDFile(sp)
					print("* done writing all CoNLL file(s); now run ./scorer.pl to evaluate our predictions")