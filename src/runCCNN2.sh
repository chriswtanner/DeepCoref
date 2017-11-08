#!/bin/bash
echo $CUDA_HOME
export PYTHONIOENCODING=UTF-8

# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"

stoppingPoints=(0.26 0.28 0.301 0.32 0.34 0.37 0.39 0.401 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.501 0.51 0.52 0.53 0.55 0.57 0.601)
source ~/researchcode/DeepCoref/venv/bin/activate

if [ ${me} = "ctanner" ]
then
	echo "[ ON BROWN NETWORK ]"
	baseDir=${brownDir}
	refDir=${refDirBrown}
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
refDir=${scriptDir}"reference-coreference-scorers-8.01/"
corpusPath=${baseDir}"data/ECB_$1/"
replacementsFile=${baseDir}"data/replacements.txt"
allTokens=${baseDir}"data/allTokensFull.txt"

hddcrpBaseFile=${10}
hddcrpFullFile=${baseDir}"data/"${hddcrpBaseFile}".WD.semeval.txt" # MAKE SURE THIS IS WHAT YOU WANT (gold vs predict)
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"
resultsDir=${baseDir}"results/"

# glove params
gWindowSize=6
embeddingsBaseFile=$9
numEpochs=50
gloveOutput=${baseDir}"data/gloveEmbeddings."${embeddingsBaseFile}".txt"

# additional coref engine params
mentionsFile=${baseDir}"data/goldTruth_events.txt"
embeddingsFile=${gloveOutput}
embeddingsType="type"
device=$2
numLayers=$3
numEpochs=$4
windowSize=$5
numNegPerPos=$6
batchSize=$7
shuffleTraining=$8
dropout=${11}
clusterMethod=${12}
numFilters=${13}
filterMultiplier=${14}
featurePOS=${15}
posType=${16}
posEmbeddingsFile=${baseDir}"data/posEmbeddings100.txt"

lemmaType=${17}
lemmaBaseFile=${18}
lemmaEmbeddingsFile=${baseDir}"data/lemmaEmbeddings."${lemmaBaseFile}".txt" # 6B.300 or 840B.300 or 400
stanOutputDir=${baseDir}"data/stanford_output/"
cd $scriptDir

echo "-------- params --------"
echo "corpus:" $1
echo "device:" ${device}
echo "numLayers:" $numLayers
echo "numEpochs:" $numEpochs
echo "windowSize:" $windowSize
echo "numNegPerPos:" $numNegPerPos
echo "batchSize:" $batchSize
echo "shuffleTraining:" $shuffleTraining
echo "embeddingsFile:" $embeddingsFile
echo "hddcrpFullFile:" $hddcrpFullFile
echo "dropout:" $dropout
echo "clusterMethod:" $clusterMethod
echo "numFilters:" $numFilters
echo "lemmaType:" $lemmaType
echo "lemmaEmbeddingsFile:" ${lemmaEmbeddingsFile}
echo "------------------------"

python3 -u CorefEngine.py --resultsDir=${resultsDir} --device=${device} \
--numLayers=${numLayers} --corpusPath=${corpusPath} --replacementsFile=${replacementsFile} \
--stitchMentions=${stitchMentions} --mentionsFile=${mentionsFile} \
--embeddingsBaseFile=${embeddingsBaseFile} --embeddingsFile=${embeddingsFile} \
--embeddingsType=${embeddingsType} --numEpochs=${numEpochs} --verbose=${verbose} \
--windowSize=${windowSize} --shuffleTraining=${shuffleTraining} --numNegPerPos=${numNegPerPos} \
--batchSize=${batchSize} \
--hddcrpBaseFile=${hddcrpBaseFile} --hddcrpFullFile=${hddcrpFullFile} \
--dropout=${dropout} --clusterMethod=${clusterMethod} \
--numFilters=${numFilters} --filterMultiplier=${filterMultiplier} \
--stanOutputDir=${stanOutputDir} \
--featurePOS=${featurePOS} --posType=${posType} --posEmbeddingsFile=${posEmbeddingsFile} \
--lemmaType=${lemmaType} --lemmaBaseFile=${lemmaBaseFile} --lemmaEmbeddingsFile=${lemmaEmbeddingsFile}

cd ${refDir}
goldFile=${baseDir}"data/gold.WD.semeval.txt"
shopt -s nullglob

for sp in "${stoppingPoints[@]}"
do
	f=${baseDir}"results/"${hddcrpBaseFile}"_lb"${lemmaBaseFile}"_nl"${numLayers}"_ne"${numEpochs}"_ws"${windowSize}"_neg"${numNegPerPos}"_bs"${batchSize}"_sFalse_e"${embeddingsBaseFile}"_dr"${dropout}"_cm"${clusterMethod}"_nf"${numFilters}"_fm"${filterMultiplier}"_fpos"${featurePOS}"_pt"${posType}"_lt"${lemmaType}"_sp"${sp}".txt"

	muc=`./scorer.pl muc ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	bcub=`./scorer.pl bcub ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	ceafe=`./scorer.pl ceafe ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	sum=`echo ${muc}+${bcub}+${ceafe} | bc`
	avg=`echo "scale=2;$sum/3.0" | bc`
	echo "CoNLLF1:" ${f} ${avg}
	rm -rf ${f}
done

# writes GloVe embeddings from the parsed corpus' output ($allTokens)
# cd "/Users/christanner/research/libraries/GloVe-master"
# ./demo.sh ${allTokens} ${gWindowSize} ${embeddingSize} ${numEpochs} ${gloveOutput}
# ./home/jrasley/set_cuda8_cudnn6.sh
# export CUDA_HOME=/contrib/projects/cuda8.0
# export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# export PATH=${CUDA_HOME}/bin:${PATH}