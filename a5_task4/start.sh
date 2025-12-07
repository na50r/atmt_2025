#!/bin/bash
for i in 3 5 10 15 20; do
    file="a5_task4_beam${i}.sh"
    sbatch $file
    sleep 5
done
