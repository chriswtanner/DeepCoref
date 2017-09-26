#!/bin/bash

# manually set these base params
scriptDir="/Users/christanner/research/DeepCoref/src/"
corpusPath="/Users/christanner/research/DeepCoref/data/ECB_SMALL/"
replacementsFile="/Users/christanner/research/DeepCoref/data/replacements.txt"
allTokens="/Users/christanner/research/DeepCoref/data/allTokensFull.txt"
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"

# glove param
windowSize=10
embeddingSize=50
numEpochs=50
gloveOutput="/Users/christanner/research/DeepCoref/data/gloveEmbeddings"

# additional coref engine params
embeddingsFile=${gloveOutput}
embeddingsType="type"

cd $scriptDir

# parses corpus and outputs a txt file, with 1 sentence per line, which is used for (1) creating embeddings; (2) stanfordCoreNLP to annotate
# python3 WriteSentencesToFile.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --outputFile=${allTokens} --verbose=${verbose}

# writes GloVe embeddings from the parsed corpus' output ($allTokens)
# cd "/Users/christanner/research/libraries/GloVe-master"
# ./demo.sh ${allTokens} ${windowSize} ${embeddingSize} ${numEpochs} ${gloveOutput}

python3 CorefEngine.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --embeddingsFile=${embeddingsFile} --embeddingsType=${embeddingsType} --verbose=${verbose}

exit 1

# runs stanfordCoreNLP, which annotates our corpus
cd ${stanfordPath}
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${allTokens} -tokenize.options untokenizable=noneKeep -parse.debug

cd ${scriptDir}
python3 AlignWithStanford.py --corpusPath=${corpusPath} --stanfordFile=${stanfordPath}allTokens1.txt.xml --replacementsFile=${replacementsFile} --verbose=t