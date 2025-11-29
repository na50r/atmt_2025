#!/bin/bash
for i in 1 3 5; do
    file="a5_task3_beam_${i}.sh"
    sbatch $file
    sleep 10
done