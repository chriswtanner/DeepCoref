#!/bin/bash
cd /Users/christanner/research/libraries/reference-coreference-scorers-8.01
goldFile="/Users/christanner/research/DeepCoref/data/gold.WD.semeval.txt"
shopt -s nullglob
files=(/Users/christanner/research/DeepCoref/results/predict*)
for f in "${files[@]}"
do
	muc=`./scorer.pl muc ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	bcub=`./scorer.pl bcub ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	ceafe=`./scorer.pl ceafe ${goldFile} ${f} | grep "Coreference: Recall" | cut -d" " -f 11 | sed 's/.$//'`
	sum=`echo ${muc}+${bcub}+${ceafe} | bc`
	avg=`echo "scale=2;$sum/3.0" | bc`
	echo ${f} ${avg}
done