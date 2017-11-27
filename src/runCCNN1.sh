#!/bin/bash
cd /home/christanner/researchcode/DeepCoref/src/
hn=`hostname`

numLayers=(2) # 3) # 1 3
numEpochs=(3) # 20)
windowSize=(0)
numNeg=(7)
batchSize=(64) # 128) # 64 128
shuffle=(f) # t
poolType=("max") # "avg")
embeddingsBaseFile=("840B.300") # 6B.300") # "840B.300")
dropout=(0.0) # 0.2 0.4)
clusterMethod=("avg")
numFilters=(32)
filterMultiplier=(2.0) # 2.0)
hddcrpBaseFile=("predict.ran")
featurePOS=("none") # none   onehot   emb_random   emb_glove
posType=("none") # none  sum  avg
lemmaType=("sum") # "sum" "avg")
dependencyType=("none") # # "sum" "avg")
charType=("concat") # "none" "concat" "sum" "avg"
SSType=("none") # "none" "sum" "avg"
SSwindowSize=(0) # 3 5 7
SSvectorSize=(0) #100 400 800)
SSlog=("True")
devDir=(6633) # 2 3 4 5 6 7 8 9 10 11 12 13 14 16 18 19 20 21 22 23 24 25)
penalty=(1) # -2 -1 1 2
activation=("relu") # "relu" "sigmoid"

source ~/researchcode/DeepCoref/venv/bin/activate
# source ~/researchcode/DeepCoref/oldcpu/bin/activate
# source /data/people/christanner/tfcpu/bin/activate

# GPU runs
for nl in "${numLayers[@]}"
do
	for pool in "${poolType[@]}"
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
														for dt in "${dependencyType[@]}"
														do
															for ct in "${charType[@]}"
															do
																for st in "${SSType[@]}"
																do
																	for ws2 in "${SSwindowSize[@]}"
																	do
																		for vs in "${SSvectorSize[@]}"
																		do
																			for sl in "${SSlog[@]}"
																			do
																				for emb in "${embeddingsBaseFile[@]}"
																				do
																					for hdd in "${hddcrpBaseFile[@]}"
																					do
																						for dd in "${devDir[@]}"
																						do
																							for pen in "${penalty[@]}"
																							do
																								for act in "${activation[@]}"
																								do
																									# qsub -pe smp 8 -l vlong -o
																									fout=gpu${hdd}_nl${nl}_pool${pool}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}_dr${dr}_cm${cm}_nf${nf}_fm${fm}_fp${fpos}_pt${pt}_lt${lt}_dt${dt}_ct${ct}_st${st}_ws2${ws2}_vs${vs}_sl${sl}_dd${dd}_p${pen}_a${act}.out
																									echo ${fout}
																									if [ ${hn} = "titanx" ] || [ ${hn} = "Christophers-MacBook-Pro-2.local" ]
																									then
																										echo "* kicking off runCCNN2 natively"
																										./runCCNN2.sh FULL gpu ${nl} ${pool} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hdd} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${dt} ${ct} ${st} ${ws2} ${vs} ${sl} ${dd} ${pen} ${act}  # > ${fout}												
																									else
																										qsub -l gpus=1 -o ${fout} runCCNN2.sh FULL gpu ${nl} ${pool} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hdd} ${dr} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${dt} ${ct} ${st} ${ws2} ${vs} ${sl} ${dd} ${pen} ${act}
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
								done
							done
						done
					done
				done
			done
		done
	done
done

