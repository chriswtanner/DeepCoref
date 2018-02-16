import sys  
import params
import os.path
import random
from ECBParser import *
from ECBHelper import *
from HDDCRPParser import *
from StanParser import *
from CCNN import *
from FFNNWD import *
from FFNNCD import *
from FFNNCDDisjoint import *
from get_coref_metrics import *
from collections import defaultdict
from sortedcontainers import SortedDict

# 2-STAGE COREF: do WD first, then merge those clusters across docs

# Coreference Resolution System for Events (uses ECB+ corpus)
class CorefEngine:
	if __name__ == "__main__":

		runFFNN = False # if False, we will use Agglomerative Cluster
		stoppingPoints = [0.501] #[0.301,0.35,0.401,0.45,0.475,0.501,0.525,0.55] #,0.501,0.55] #,0.55]                                                                   
		stoppingPoints2 = [0.501,0.525] #[0.45,0.47,0.501,0.525,0.55,0.575,0.601,0.625,0.65,0.675,0.701]
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

		# trains and tests the pairwise-predictions via Conjoined-CNN
		wd_ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed, True) # creates WD-CCNN model
		(wd_dev_pairs, wd_dev_preds, wd_testing_pairs, wd_testing_preds) = wd_ccnnEngine.trainAndTest()

		cd_ccnnEngine = CCNN(args, corpus, helper, hddcrp_parsed, False) # creates CD-CCNN model
		(cd_dev_pairs, cd_dev_preds, cd_testing_pairs, cd_testing_preds) = cd_ccnnEngine.trainAndTest()

		# creates output label, for displaying results clearly
		outputLabel = ""
		if args.useECBTest:
			outputLabel += "ECBTest"
		else:
			outputLabel += "HDDCRPTest"
		if runFFNN:
			outputLabel += "FFNN"
		else:
			outputLabel += "AGG"

		# perform WD first, then CD
		bestTestSP = -1
		bestTestSP2 = -1
		bestTestF1 = -1
		for sp in stoppingPoints:

			# performs WD via Agglomerative (ECB Test Mentions)
			if args.useECBTest: # use corpus' test mentions
				(wd_predictedClusters, wd_goldenClusters) = wd_ccnnEngine.aggClusterPredictions(wd_testing_pairs, wd_testing_preds, sp)
				(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(wd_goldenClusters, wd_predictedClusters)
				print("ECBTest AGG WD F1 sp:",str(sp),"=",str(conll_f1),"MUC:",str(muc_f1),"BCUB:",str(bcub_f1),"CEAF:",str(ceafe_f1))
			
			# performs WD via Agglomerative (HDDCRP Test Mentions)
			else:
				wd_predictedClusters = helper.clusterHPredictions(wd_testing_pairs, wd_testing_preds, sp, True)
				wd_ccnnEngine.analyzeResults(wd_testing_pairs, wd_testing_preds, wd_predictedClusters)
				helper.writeCoNLLFile(wd_predictedClusters, sp)

			# perform CD
			for sp2 in stoppingPoints2:

				# ECB Test Mentions
				if args.useECBTest:
					cd_predictedClusters = None
					cd_goldenClusters = None
					
					# ECBTest; FFNN to cluster (NOT IMPLEMENTED; LOWEST PRIORITY)
					if runFFNN:
						print("[ECBTest: FFNN Mode]")
						ffnnEngine = FFNNCDDisjoint(args, corpus, helper, hddcrp_parsed, cd_dev_pairs, cd_dev_preds, cd_testing_pairs, cd_testing_preds)
						ffnnEngine.train()
						(cd_predictedClusters, cd_goldenClusters) = ffnnEngine.clusterWDClusters(sp2)
						# we actually use goldenClusters because it's the ECBTest (gold)
					
					# ECBTest; Agglomerative (WORKS)
					else:
						print("[ECBTest: AGG Mode]")
						(cd_predictedClusters, cd_goldenClusters) = cd_ccnnEngine.aggClusterWDClusters(wd_predictedClusters, cd_testing_pairs, cd_testing_preds, sp2)
					
					print("# goldencluster:",str(len(cd_goldenClusters)),"# predicted:",str(len(cd_predictedClusters)))		
					(bcub_p, bcub_r, bcub_f1, muc_p, muc_r, muc_f1, ceafe_p, ceafe_r, ceafe_f1, conll_f1) = get_conll_scores(cd_goldenClusters, cd_predictedClusters)
					print("ECBTest - CD F1 sp2:",str(sp2),"=",str(conll_f1),"MUC:",str(muc_f1),"BCUB:",str(bcub_f1),"CEAF:",str(ceafe_f1))
					if conll_f1 > bestTestF1:
						bestTestF1 = conll_f1
						bestTestSP = sp
						bestTestSP2 = sp2
				
				# HDDCRP's test mentions
				else:
					cd_predictedClusters = None
					# HDDCRP Test; FFNN (HIGHEST PRIORITY TO IMPLEMENT)
					if runFFNN:
						print("[HDDCRPTest: FFNN Mode]")
						ffnnEngine = FFNNCDDisjoint(args, corpus, helper, hddcrp_parsed, cd_dev_pairs, cd_dev_preds, cd_testing_pairs, cd_testing_preds)
						ffnnEngine.train()
						(cd_predictedClusters, _) = ffnnEngine.clusterWDClusters(sp2)
						# we ignore goldenClusters because that isn't the gold Truth

					# HDDCRP Test; Agglomerative (WORKS)
					else:
						print("[HDDCRPTest: AGG Mode]")
						cd_predictedClusters = helper.clusterWDHPredictions(wd_predictedClusters, cd_testing_pairs, cd_testing_preds, sp2)

					helper.writeCoNLLFile(cd_predictedClusters, sp, sp2)
					helper.convertWDFileToCDFile(sp, sp2)

		if args.useECBTest:
			print("[FINAL",outputLabel,"RESULTS]: BEST SP",str(bestTestSP),"SP2:",str(bestTestSP2),"F1:",str(bestTestF1))