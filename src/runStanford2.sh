#!/bin/bash
source ~/researchcode/DeepCoref/venv/bin/activate

stanfordPath=$1
inputFile=$2
outputDir=$3
echo "-------- params --------"
echo "stanfordPath:" ${stanfordPath}
echo "inputFile:" ${inputFile}
echo "outputDir:" ${outputDir}
echo "------------------------"

# runs stanfordCoreNLP, which annotates our corpus
cd ${stanfordPath}
java -cp "*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${inputFile} -outputDirectory ${outputDir} -tokenize.options untokenizable=noneKeep -parse.debug

