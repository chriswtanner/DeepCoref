#!/bin/bash
corpusPath="/Users/christanner/research/DeepCoref/data/ECB_SMALL/"
replacementsFile="/Users/christanner/research/DeepCoref/data/replacements.txt"
allTokens="/Users/christanner/research/DeepCoref/data/allTokens.txt"
verbose="true"

stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09"


python WriteSentencesToFile.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --outputFile=${allTokens} --verbose=${verbose}

#cd ${stanfordPath}
#java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${allTokens}

