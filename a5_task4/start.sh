#!/bin/bash
OUT_DIR=~/shares/groups/turing/a5/a5_task4/translations
rm -rf $OUT_DIR
mkdir $OUT_DIR

for i in 3 5 10 15 20; do
    file="a5_task4_beam${i}.sh"
    sbatch $file
    sleep 5
done
