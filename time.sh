#!/bin/bash
out="times.txt"

echo "Translation times extracted:" >> "$out"
date_stamp=$(date)
echo "Date: $date_stamp" >> "$out" 
curr=$(git branch --show-current)
echo "Branch: $curr" > "$out" 
for i in 1 3 5; do
    file="a5_test_beam_${i}.out"
    grep "Translation" "$file" >> "$out"
done
echo "" >> "$out"


