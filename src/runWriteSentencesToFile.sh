#!/bin/bash
export PYTHONIOENCODING=UTF-8

# NOTE: this will always be run locally, as it only takes a minute
# USAGE: ./runWriteSentencesToFile.sh <corpus>, where <corpus> is of {TINY, TEST, SMALL, HALF, FULL}
baseDir="/Users/christanner/research/DeepCoref/"
scriptDir=${baseDir}"src/"
corpusPath=${baseDir}"data/ECB_$1/"
replacementsFile=${baseDir}"data/replacements.txt"
writeOutDir=${baseDir}"data/parsed/"
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"

cd $scriptDir
echo "-------- params --------"
echo "corpus:" $1
echo "replacementsFile:" $replacementsFile
echo "stitchMentions:" $stitchMentions
echo "writeOutDir:" $writeOutDir
echo "------------------------"

# parses corpus and outputs a txt file, with 1 sentence per line, which is used for (1) creating embeddings; (2) stanfordCoreNLP to annotate
python3 WriteSentencesToFile.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --writeOutDir=${writeOutDir} --verbose=${verbose}
