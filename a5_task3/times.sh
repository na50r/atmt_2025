#!/bin/bash
out="times.txt"
date_stamp=$(date)
echo "Date: $date_stamp" >> $out 
curr=$(git branch --show-current)
echo "Branch: $curr" >> $out
for i in 1 3 5; do
    file="a5_task3_beam${i}.out"
    time_d=$(grep "Translation" $file | grep -Po "\S+ seconds")
    bleu=$(grep 'BLEU' $file)
    echo $file $time_d >> $out
    echo $file $bleu >> $out
done
echo "" >> $out