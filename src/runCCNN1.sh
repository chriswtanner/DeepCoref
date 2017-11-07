#!/bin/bash
cd /home/christanner/researchcode/DeepCoref/src/
hn=`hostname`

numLayers=(2) # 3) # 1 3
numEpochs=(1) # 25) # 30) # 5 10 20
windowSize=(0) # 1 2 3
numNeg=(7) # 10) # 5 10 15
batchSize=(128) # 64 128
shuffle=(f) # t                                                                                  
embSize=(400) # 50                                                                                    
dropout=(0.3) # 0.4) # 0.0 0.1 .2 .3 .5
numFilters=(64) # 128) # 300 600)
filterMultiplier=(1.0) # 1.5 2.0)
hddcrpBaseFile=("predict" "predict.ran")
clusterMethod=("avgavg") # "avg" "avgavg") # "min" "avg"
featurePOS=("none") # none   onehot   emb_random   emb_glove
posType=("none") # none  sum  avg
lemmaType=("avg") # "sum" "avg")
lemmaBaseFile=("400" "6B.300") # 6B.300 or 840B.300 or 400
source ~/researchcode/DeepCoref/venv/bin/activate
# source ~/researchcode/DeepCoref/oldcpu/bin/activate
# source /data/people/christanner/tfcpu/bin/activate

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
														for lb in "${lemmaBaseFile[@]}"
														do
															for hdd in "${hddcrpBaseFile[@]}"
															do
																# qsub -pe smp 8 -l vlong -o
																fout=gpuGOLD_lb${lb}_nl${nl}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}_dr${dr}_cm${cm}_nf${nf}_fm${fm}_${fpos}_${pt}_lt${lt}.out
																echo ${fout}
																if [ ${hn} = "titanx" ]
																then
																	./runCCNN2.sh FULL gpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hddcrp} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${lb} > ${fout}												
																else
																	qsub -l gpus=1 -o ${fout} runCCNN2.sh FULL gpu ${nl} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hddcrp} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${lb}
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
	done
done

