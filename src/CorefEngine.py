import sys  
import params
from ECBParser import *
from ECBHelper import *
from SiameseCNN import *
from get_coref_metrics import *
# parses the corpus and runs Coref Resoultion on the mentions
class CorefEngine:
	if __name__ == "__main__":

		# handles passed-in args
		args = params.setCorefEngineParams()

		# parses corpus
		corpus = ECBParser(args)

		# constructs helper class
		helper = ECBHelper(corpus, args)
		
		#response = helper.constructCoNLLClustersFromFile("/Users/christanner/research/DeepCoref/results/test_hddcrp2.response")
		#print(str(len(response)))
		#helper.constructCoNLLTestFileCD("/Users/christanner/research/DeepCoref/results/test_cd.keys")
		#helper.constructCoNLLTestFileWD("/Users/christanner/research/DeepCoref/results/test_wd.keys")

		# trains and tests the pairwise-predictions
		corefEngine = SiameseCNN(args, corpus, helper)
		(pairs, predictions) = corefEngine.run()

		stoppingPoints = [0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.75]
		f1s = []
		for sp in stoppingPoints:
			(predictedClusters, goldenClusters) = corefEngine.clusterPredictions(pairs, predictions, sp)
			f1s.append(get_conll_f1(goldenClusters, predictedClusters))
		for i in zip(stoppingPoints,f1s):
			print(i)

		#helper.evaluateCoNLL(predictedClusters, goldenClusters)

