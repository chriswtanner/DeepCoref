#!/bin/bash
prefix="devA" # used to help identify experiments' outputs, as the output files will have this prefix
featureMap=(2) # 4)
numLayers=(1) # 3) # 1 3
numEpochs=(3) # 20)
windowSize=(0)
numNeg=(2)
batchSize=(128) # 128) # 64 128
shuffle=(f) # t
poolType=("max") # "avg")
embeddingsBaseFile=("6B.300") # 6B.300") # "840B.300")
dropout=(0.0) # 0.2 0.4)
CCNNOpt=("adam") # "rms" "adam" "adagrad"
clusterMethod=("min")
numFilters=(16)
filterMultiplier=(1.0) # 2.0)
hddcrpBaseFile=("predict.ran")
featurePOS=("none") # none   onehot   emb_random   emb_glove
posType=("none") # none  sum  avg
lemmaType=("sum") # "sum" "avg")
dependencyType=("none") # # "sum" "avg")
charType=("none") # "none" "concat" "sum" "avg"
SSType=("none") # "none" "sum" "avg"
SSwindowSize=(0) # 3 5 7
SSvectorSize=(0) #100 400 800)
SSlog=("True")
devDir=(23) # this # and above will be the dev dirs.  See ECBHelper.py for more

cd /home/christanner/researchcode/DeepCoref/src/
hn=`hostname`

# FEATURE MAP OVERRIDE
if [[ " ${featureMap[*]} " == *"1"* ]]; then
	featurePOS=("emb_glove")
	posType=("sum")
fi
if [[ " ${featureMap[*]} " == *"2"* ]]; then
	lemmaType=("sum")
fi
if [[ " ${featureMap[*]} " == *"3"* ]]; then
	dependencyType=("sum")
fi
if [[ " ${featureMap[*]} " == *"4"* ]]; then
	charType=("concat")
fi
if [[ " ${featureMap[*]} " == *"5"* ]]; then
	SSType=("sum")
	SSwindowSize=(5)
	SSvectorSize=(400)
fi

# FFNN params
FFNNnumEpochs=(10)
FFNNnumCorpusSamples=(1) # 5 10 20)
FFNNPosRatio=(0.8) # 0.2 0.8
FFNNOpt=("adam") # "rms" "adam" "adagrad"
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
								for co in "${CCNNOpt[@]}"
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
																								for fn in "${FFNNnumEpochs[@]}"
																								do
																									for fp in "${FFNNnumCorpusSamples[@]}"
																									do
																										for fo in "${FFNNOpt[@]}"
																										do
																											# qsub -pe smp 8 -l vlong -o
																											fout=gpu_${prefix}_${hdd}_nl${nl}_pool${pool}_ne${ne}_ws${ws}_neg${neg}_bs${bs}_s${s}_e${emb}_dr${dr}_co${co}_cm${cm}_nf${nf}_fm${fm}_fp${fpos}_pt${pt}_lt${lt}_dt${dt}_ct${ct}_st${st}_ws2${ws2}_vs${vs}_sl${sl}_dd${dd}_fn${fn}_fp${fp}_fo${fo}.out
																											echo ${fout}
																											if [ ${hn} = "titanx" ] || [ ${hn} = "Christophers-MacBook-Pro-2" ]
																											then
																												echo "* kicking off runCCNN2 natively"
																												./runCCNN2.sh FULL gpu ${nl} ${pool} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hdd} ${dr} ${co} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${dt} ${ct} ${st} ${ws2} ${vs} ${sl} ${dd} ${fn} ${fp} ${fo} # > ${fout}												
																											else
																												qsub -l gpus=1 -o ${fout} runCCNN2.sh FULL gpu ${nl} ${pool} ${ne} ${ws} ${neg} ${bs} ${s} ${emb} ${hdd} ${dr} ${co} ${cm} ${nf} ${fm} ${fpos} ${pt} ${lt} ${dt} ${ct} ${st} ${ws2} ${vs} ${sl} ${dd} ${fn} ${fp} ${fo}
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
	done
done

