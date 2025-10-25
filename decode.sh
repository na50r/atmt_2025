#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=decode.out

python decode_binary_dataset.py \
  --binary-dir ./data-bin \
  --output-dir ./decoded \
  --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
  --source-lang cz \
  --target-lang en 