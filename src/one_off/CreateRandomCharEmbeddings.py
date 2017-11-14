import random
class CreateRandomCharEmbeddings:
	inputFile = "/Users/christanner/research/DeepCoref/data/charGloveEmbeddings.txt"
	outputFile = "/Users/christanner/research/DeepCoref/data/charRandomEmbeddings.txt"

	chars = set()
	length = 20
	f = open(inputFile, 'r', encoding="utf-8")
	for line in f:
		chars.add(line.rstrip().split(" ")[0])
	f.close()

	print("# chars:",str(len(chars)))

	fout = open(outputFile, 'w')
	for c in chars:
		fout.write(str(c))
		for i in range(length):
			fout.write(" " + str(float(random.uniform(-1,1)))[0:8])
		fout.write("\n")
	fout.close()