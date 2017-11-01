#!/bin/bash
echo $CUDA_HOME
export PYTHONIOENCODING=UTF-8

# USAGE: ./runStanford.sh
# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"

stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"

source ~/researchcode/DeepCoref/venv/bin/activate

if [ ${me} = "ctanner" ]
then
	echo "[ ON BROWN NETWORK ]"
	baseDir=${brownDir}
	stanfordPath="/home/christanner/researchcode/libraries/stanford-corenlp-full-2017-06-09/"
fi

shopt -s nullglob
corpusPath=${baseDir}"data/parsed/"
outputDir=${baseDir}"data/stanford_output/"
files=(${corpusPath}*)

echo "-------- params --------"
echo "corpusPath:" ${corpusPath}
echo "outputDir:" ${outputDir}
echo "------------------------"
for f in "${files[@]}"
do
	echo "file:" ${f}
	echo "writing log file:" stan_${filename}
	filename="${f%.*}"
	filename="${filename##*/}"
	qsub -l short -o stan_${filename}.out runStanford2.sh ${stanfordPath} ${f} ${outputDir}
done


# runs stanfordCoreNLP, which annotates our corpus
# cd ${stanfordPath}
# java -cp "*" -Xmx4g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${inputFile} -outputDirectory ${outputDir} -tokenize.options untokenizable=noneKeep -parse.debug

