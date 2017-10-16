#!/bin/bash
echo $CUDA_HOME
export PYTHONIOENCODING=UTF-8
# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"
# ./home/jrasley/set_cuda8_cudnn6.sh
# export CUDA_HOME=/contrib/projects/cuda8.0
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# export PATH=${CUDA_HOME}/bin:${PATH}
source ~/researchcode/DeepCoref/venv/bin/activate

if [ ${me} = "ctanner" ]
then
	echo "[ ON BROWN NETWORK ]"
	baseDir=${brownDir}
	echo $CUDA_HOME
	if [ ${hn} = "titanx" ]
	then
		echo "*   ON TITAN!"
		export CUDA_HOME=/usr/local/cuda/
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
		export PATH=${CUDA_HOME}/bin:${PATH}
		echo ${CUDA_HOME}
		echo ${LD_LIBRARY_PATH}
	else
		echo "*   ON THE GRID!"
		# export CUDA_HOME=/usr
		export CUDA_HOME=/contrib/projects/cuda8.0
		export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
		export PATH=${CUDA_HOME}/bin:${PATH}
    	echo ${CUDA_HOME}
		echo ${LD_LIBRARY_PATH}
	fi 
fi

scriptDir=${baseDir}"src/"
corpusPath=${baseDir}"data/ECB_FULL/"
replacementsFile=${baseDir}"data/replacements.txt"
allTokens=${baseDir}"data/allTokensFull.txt"
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"

# glove param
gWindowSize=6
embeddingSize=$8
numEpochs=50
gloveOutput=${baseDir}"data/gloveEmbeddings"${embeddingSize}".txt"

# additional coref engine params
mentionsFile=${baseDir}"data/goldTruth_events.txt"
embeddingsFile=${gloveOutput}
embeddingsType="type"
device=$1
numLayers=$2
numEpochs=$3
windowSize=$4
numNegPerPos=$5
batchSize=$6
shuffleTraining=$7
cd $scriptDir
echo ${device}
# parses corpus and outputs a txt file, with 1 sentence per line, which is used for (1) creating embeddings; (2) stanfordCoreNLP to annotate
#python3 WriteSentencesToFile.py --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --outputFile=${allTokens} --verbose=${verbose} --mentionsFile=${mentionsFile}

# writes GloVe embeddings from the parsed corpus' output ($allTokens)
# cd "/Users/christanner/research/libraries/GloVe-master"
# ./demo.sh ${allTokens} ${gWindowSize} ${embeddingSize} ${numEpochs} ${gloveOutput}
# if [ ${device} = "cpu" ]
# then
#	export CUDA_VISIBLE_DEVICES=
# fi
python3 -u CorefEngine.py --device=${device} --numLayers=${numLayers} --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} --stitchMentions=${stitchMentions} --mentionsFile=${mentionsFile} --embeddingsFile=${embeddingsFile} --embeddingsType=${embeddingsType} --numEpochs=${numEpochs} --verbose=${verbose} --windowSize=${windowSize} --shuffleTraining=${shuffleTraining} --numNegPerPos=${numNegPerPos} --batchSize=${batchSize}

exit 1

# runs stanfordCoreNLP, which annotates our corpus
cd ${stanfordPath}
java -cp "*" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ${allTokens} -tokenize.options untokenizable=noneKeep -parse.debug

cd ${scriptDir}
python3 AlignWithStanford.py --corpusPath=${corpusPath} --stanfordFile=${stanfordPath}allTokens1.txt.xml --replacementsFile=${replacementsFile} --verbose=t --mentionsFile=${mentionsFile}
