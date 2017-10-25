import operator
from collections import defaultdict
inputFile = "/Users/christanner/research/hddcrp_results/fields.csv"

counts = defaultdict(int)
f = open(inputFile, 'r')
for line in f:
	counts[line.rstrip()] += 1
f.close()

sorted_x = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
for s in sorted_x:
	print(s)