#!/usr/bin/bash

thresholds="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
beam_sizes="3 5 10 14 20"

for t in $thresholds; do
    for b in $beam_sizes; do
        echo "Submitting job: threshold=$t, beam_size=$b"
        sbatch --job-name=beam${b}_thr${t} run.sh $t $b
        sleep 5
    done
done

