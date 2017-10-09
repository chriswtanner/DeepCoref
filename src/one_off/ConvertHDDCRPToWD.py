import sys  
from collections import defaultdict

# PURPOSE: our HDDCRP's output is for all links (coref), 
# 	so we wish to also look at just the WD performance
#	(to confirm the paper's findings)
responseFile="/Users/christanner/research/DeepCoref/results/test_hddcrp2.response"
outputWDFile="/Users/christanner/research/DeepCoref/results/test_hddcrp_wd.response"

f = open(responseFile, 'r')
f.readline()
clusterToDMs = defaultdict(list)
for line in f:
	line = line.rstrip()
	if line == "#end document":
		break
	_, dm, clusterID = line.rstrip().split()
	clusterToDMs[clusterID].append(dm)
f.close()

# writes WD file
f = open(outputWDFile, 'w')
f.write("#begin document (t);\n")
clusterNum = 0
for clusterID in clusterToDMs.keys():
	docsSoFar = {}
	for dm in clusterToDMs[clusterID]:
		doc_id = dm.split(";")[0]
		dirNum = dm[0:dm.find("_")]
		if doc_id not in docsSoFar:
			docsSoFar[doc_id] = clusterNum + 1
			clusterNum += 1
		curNum = docsSoFar[doc_id]
		f.write(str(dirNum) + "\t" + str(dm) + \
			"\t(" + str(curNum) + ")\n")
f.write("#end document (t);\n")
