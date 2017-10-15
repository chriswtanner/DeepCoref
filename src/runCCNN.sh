#!/bin/bash
cd /home/christanner/researchcode/DeepCoref/src/
numLayers=(2) 
numEpochs=(3) # 5 25 50)
windowSize=(2) # 5)
numNeg=(2) # 5 10)
batchSize=(28) # 128)
shuffle=(f) # t)
embSize=(400) # 50
source ~/researchcode/DeepCoref/venv/bin/activate
# source ~/researchcode/DeepCoref/oldcpu/bin/activate
# source /data/people/christanner/tfcpu/bin/activate

for nl in "${numLayers[@]}"
do
    for ne in "${numEpochs[@]}"
    do
	for ws in "${windowSize[@]}"
	do
	    for neg in "${numNeg[@]}"
	    do
		for bs in "${batchSize[@]}"
		do
		    for s in "${shuffle[@]}"
		    do
			for emb in "${embSize[@]}"
			do
			    # qsub -pe smp 32 -l vlong -o cpu32b_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}.out runCoref.sh cpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb}
			    a=1
			done
		    done
		done
	    done
	done
    done
done
# exit 1
# GPU
for nl in "${numLayers[@]}"
do
    for ne in "${numEpochs[@]}"
    do
	for ws in "${windowSize[@]}"
	do
	    for neg in "${numNeg[@]}"
	    do
		for bs in "${batchSize[@]}"
		do
		    for s in "${shuffle[@]}"
		    do
			for emb in "${embSize[@]}"
			do
			    qsub -l gpus=1 -o gpuB_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}.out runCoref.sh gpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb}
			done
		    done
		done
	    done
	done
    done
done

