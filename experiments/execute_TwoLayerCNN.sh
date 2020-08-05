#!/usr/bin/env bash
cd "$(dirname $0)/.." 
seed=(1 2)
std=(3 5 7 9 11 13)
n=${#seed[@]}
m=${#std[@]}

for ((i=0;i<$n;++i))
do	
    for ((j=0;j<$m;++j))
    do	
    echo "run two layer CNN with seed=${seed[$i]}, std=${std[$j]}"
python3 RandInput_LRCount_TwoLayer.py --width=4 --height=1 --k_w=2 --k_h=1 -stride=1 -out_channel=3 --std=${std[$j]} -seed=${seed[$i]} --sample_N=20000000 
    done
done
 
