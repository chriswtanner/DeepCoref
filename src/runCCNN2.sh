#!/bin/bash
echo $CUDA_HOME
export PYTHONIOENCODING=UTF-8

# manually set these base params
me=`whoami`
hn=`hostname`
baseDir="/Users/christanner/research/DeepCoref/"
brownDir="/home/ctanner/researchcode/DeepCoref/"

stoppingPoints=(0.51) # (0.15 0.17 0.19 0.21 0.23 0.26 0.28 0.301 0.32 0.34 0.37 0.39 0.401 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.501 0.51 0.52 0.53 0.55 0.57 0.601)

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
		source ~/researchcode/DeepCoref/venv/bin/activate
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
charEmbeddingsFile=${baseDir}"data/charRandomEmbeddings.txt"
allTokens=${baseDir}"data/allTokensFull.txt"

hddcrpBaseFile=${11}
hddcrpFullFile=${baseDir}"data/"${hddcrpBaseFile}".WD.semeval.txt" # MAKE SURE THIS IS WHAT YOU WANT (gold vs predict)
verbose="true"
stanfordPath="/Users/christanner/research/libraries/stanford-corenlp-full-2017-06-09/"
stitchMentions="False"
resultsDir=${baseDir}"results/"

# glove params
gWindowSize=6
embeddingsBaseFile=${10}
numEpochs=50
gloveOutput=${baseDir}"data/gloveEmbeddings."${embeddingsBaseFile}".txt"

stoplistFile=${baseDir}"data/stopwords.txt"

# additional coref engine params
mentionsFile=${baseDir}"data/goldTruth_events.txt"
embeddingsFile=${gloveOutput}
embeddingsType="type"
device=$2
numLayers=$3
poolType=$4
numEpochs=$5
windowSize=$6
numNegPerPos=$7
batchSize=$8
shuffleTraining=$9
dropout=${12}
clusterMethod=${13}
numFilters=${14}
filterMultiplier=${15}
# features
featurePOS=${16}
posType=${17}
posEmbeddingsFile=${baseDir}"data/posEmbeddings100.txt"
lemmaType=${18}
dependencyType=${19}
charType=${20}
SSType=${21}
SSwindowSize=${22}
SSvectorSize=${23}
SSlog=${24}
stanOutputDir=${baseDir}"data/stanford_output/"

echo "-------- params --------"
echo "corpus:" $1
echo "stoplistFile:" $stoplistFile
echo "resultsDir:" ${resultsDir}
echo "device:" ${device}
echo "numLayers:" $numLayers
echo "poolType:" $poolType
echo "replacementsFile:" ${replacementsFile}
echo "stitchMentions:" $stitchMentions
echo "mentionsFile:" $mentionsFile
echo "embeddingsBaseFile:" $embeddingsBaseFile
echo "embeddingsFile:" $embeddingsFile
echo "embeddingsType:" $embeddingsType
echo "numEpochs:" $numEpochs
echo "verbose:" $verbose
echo "windowSize:" $windowSize
echo "shuffleTraining:" $shuffleTraining
echo "numNegPerPos:" $numNegPerPos
echo "batchSize:" $batchSize
echo "hddcrpBaseFile:" $hddcrpBaseFile
echo "hddcrpFullFile:" $hddcrpFullFile
echo "dropout:" $dropout
echo "clusterMethod:" $clusterMethod
echo "numFilters:" $numFilters
echo "filterMultiplier:" $filterMultiplier
echo "stanOutputDir:" $stanOutputDir
echo "featurePOS:" $featurePOS
echo "posType:" $posType
echo "posEmbeddingsFile:" $posEmbeddingsFile
echo "lemmaType:" $lemmaType
echo "dependencyType:" $dependencyType
echo "charEmbeddingsFile:" $charEmbeddingsFile
echo "charType:" $charType
echo "SSType:" $SSType
echo "SSwindowSize:" $SSwindowSize
echo "SSvectorSize:" $SSvectorSize
echo "SSlog:" $SSlog
echo "------------------------"

cd $scriptDir

python3 -u CorefEngine.py --resultsDir=${resultsDir} \
--stoplistFile=${stoplistFile} \
--device=${device} \
--numLayers=${numLayers} --poolType=${poolType} --corpusPath=${corpusPath} \
--replacementsFile=${replacementsFile} \
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
--lemmaType=${lemmaType} \
--dependencyType=${dependencyType} \
--charEmbeddingsFile=${charEmbeddingsFile} \
--charType=${charType} \
--SSType=${SSType} \
--SSwindowSize=${SSwindowSize} \
--SSvectorSize=${SSvectorSize} \
--SSlog=${SSlog}

exit 1 
cd ${refDir}
goldFile=${baseDir}"data/gold.WD.semeval.txt"
shopt -s nullglob

for sp in "${stoppingPoints[@]}"
do
	f=${baseDir}"results/"${hddcrpBaseFile}"_nl"${numLayers}"_pool"${poolType}"_ne"${numEpochs}"_ws"${windowSize}"_neg"${numNegPerPos}"_bs"${batchSize}"_sFalse_e"${embeddingsBaseFile}"_dr"${dropout}"_cm"${clusterMethod}"_nf"${numFilters}"_fm"${filterMultiplier}"_fp"${featurePOS}"_pt"${posType}"_lt"${lemmaType}"_dt"${dependencyType}"_ct"${charType}"_st"${SSType}"_ws2"${SSwindowSize}"_vs"${SSvectorSize}"_sl"${SSlog}"_sp"${sp}".txt"
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