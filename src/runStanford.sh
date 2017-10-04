#!/bin/bash
echo $CUDA_HOME

# manually set these base params
me=`whoami`
baseDir="/Users/christanner/research/DeepCoref/"
gridDir="/home/christanner/researchcode/DeepCoref/"
if [ ${me} = "ctanner" ]
then
    echo "ON THE GRID!"
    baseDir=${gridDir}
    #./home/jrasley/set_cuda8_cudnn6.sh
    #export CUDA_HOME=/contrib/projects/cuda8.0
    #export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
    #export PATH=${CUDA_HOME}/bin:${PATH}
    echo $CUDA_HOME
fi

scriptDir=${baseDir}"src/"
corpusPath=${baseDir}"data/ECB_SMALL/"
replacementsFile=${baseDir}"data/replacements.txt"
allTokens=${baseDir}"data/allTokensFull.txt"
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"

# glove param
gWindowSize=10
embeddingSize=50
numEpochs=50
gloveOutput=${baseDir}"data/gloveEmbeddings.txt"

# additional coref engine params
mentionsFile=${baseDir}"data/goldTruth_events.txt"
shuffleTraining="f"
embeddingsFile=${gloveOutput}
embeddingsType="type"
numEpochs=$1
windowSize=$2
numNegPerPos=$3
batchSize=$4

cd $scriptDir

# parses corpus and outputs a txt file, with 1 sentence per line, which is used for (1) creating embeddings; (2) stanfordCoreNLP to annotate
# python3 WriteSentencesToFile.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --outputFile=${allTokens} --verbose=${verbose} --mentionsFile=${mentionsFile}

# writes GloVe embeddings from the parsed corpus' output ($allTokens)
# cd "/Users/christanner/research/libraries/GloVe-master"
# ./demo.sh ${allTokens} ${gWindowSize} ${embeddingSize} ${numEpochs} ${gloveOutput}

python3 CorefEngine.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --mentionsFile=${mentionsFile} --embeddingsFile=${embeddingsFile} --embeddingsType=${embeddingsType} --numEpochs=${numEpochs} --verbose=${verbose} --windowSize=${windowSize} --shuffleTraining=${shuffleTraining} --numNegPerPos=${numNegPerPos} --batchSize=${batchSize}

exit 1

# runs stanfordCoreNLP, which annotates our corpus
cd ${stanfordPath}
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${allTokens} -tokenize.options untokenizable=noneKeep -parse.debug

cd ${scriptDir}
python3 AlignWithStanford.py --corpusPath=${corpusPath} --stanfordFile=${stanfordPath}allTokens1.txt.xml --replacementsFile=${replacementsFile} --verbose=t --mentionsFile=${mentionsFile}