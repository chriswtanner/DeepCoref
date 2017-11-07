#!/bin/bash
cd /home/christanner/researchcode/DeepCoref/src/
hn=`hostname`

numLayers=(2) # 3) # 1 3
numEpochs=(15) # 30) # 5 10 20
windowSize=(0) # 1 2 3
numNeg=(7) # 10) # 5 10 15
batchSize=(256) # 64 128
shuffle=(f) # t                                                                                            
embSize=(400) # 50                                                                                    
dropout=(0.3) # 0.4) # 0.0 0.1 .2 .3 .5
numFilters=(100) # 300 600)
filterMultiplier=(1.0)
hddcrp="predict"
clusterMethod=("min") # "avg" "avgavg") # "min" "avg"
featurePOS=("emb_glove") # none   onehot   emb_random   emb_glove
posType=("avg") # none  sum  avg
lemmaType=("avg") # "sum" "avg")
source ~/researchcode/DeepCoref/venv/bin/activate
# source ~/researchcode/DeepCoref/oldcpu/bin/activate
# source /data/people/christanner/tfcpu/bin/activate

# CPU runs	 
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
							for dr in "${dropout[@]}"
							do
								for cm in "${clusterMethod[@]}"
								do
									# qsub -pe smp 8 -l vlong -o cpuGOLD_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}_dr${dr}_cm${cm}.out runCCNN2.sh FULL cpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hddcrp} ${dr} ${cm}
									a=1
								done
							done
						done
					done
				done
			done
		done
	done
done

# GPU runs
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
							for dr in "${dropout[@]}"
							do
								for cm in "${clusterMethod[@]}"
								do
									for fpos in "${featurePOS[@]}"
									do
										for pt in "${posType[@]}"
										do
											for nf in "${numFilters[@]}"
											do
												for fm in "${filterMultiplier[@]}"
												do
													for lt in "${lemmaType[@]}"
													do
														fout=gpuGOLD_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}_dr${dr}_cm${cm}_nf${nf}_fm${fm}_${fpos}_${pt}_lt${lt}.out
														if [ ${hn} = "titanx" ]
														then
															./runCCNN2.sh FULL gpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hddcrp} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} > ${fout}												
														else
															qsub -l gpus=1 -o ${fout} runCCNN2.sh FULL gpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hddcrp} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt}
														fi
													done
												done
											done
										done
									done	
								done
							done
						done
					done
				done
			done
		done
	done
done

