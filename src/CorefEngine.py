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

		hddcrpFile="/home/christanner/researchcode/DeepCoref/results/test_hddcrp_wd.response"
		f = open(hddcrpFile, 'r')
		hddcrpDMs = set()
		f.readline()
		hddcrpClusters = defaultdict(set)
		for line in f:
			line = line.rstrip()
			if line == "#end document (t);":
				break
			_, tmp, clusterID = line.rstrip().split()
			(doc_id,m_id) = tmp.split(";")
			dm = (doc_id,int(m_id))
			hddcrpDMs.add(dm)
			c_id = clusterID[1:-1]
			hddcrpClusters[clusterID].add(dm)
		f.close()
		print("# dms in hddcrp's:", str(len(hddcrpDMs)))
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

		stoppingPoints = [0.68]
		f1s = []
		for sp in stoppingPoints:
			(predictedClusters, goldenClusters) = corefEngine.clusterPredictions(pairs, predictions, sp)
			f1s.append(get_conll_f1(goldenClusters, predictedClusters))

			goldenDMs = set()
			missingFromHDDCRP = set()
			for _ in goldenClusters.keys():
				for i in goldenClusters[_]:
					goldenDMs.add(i)
					if i not in hddcrpDMs:
						missingFromHDDCRP.add(i)

			missingFromGolden = set()
			for i in hddcrpDMs:
				if i not in goldenDMs:
					missingFromGolden.add(i)
			print("# goldenDMs:",str(len(goldenDMs)))
			print("# dms in hddcrp's:", str(len(hddcrpDMs)))
			print("# missing from hddcrp:", str(len(missingFromHDDCRP)))
			print("# missing from golden:", str(len(missingFromGolden)))
			print("hddcrp's perf:", str(get_conll_f1(goldenClusters, hddcrpClusters)))
			exit(1)
		for i in zip(stoppingPoints,f1s):
			print(i)


