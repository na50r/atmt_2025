#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=bs_exp_greedy.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE
python translate.py \
    --cuda \
    --seed 666 \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en-a3-task1-base/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en-a3-task1-base/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en-a3-task1-mqa/checkpoints/checkpoint_best.pt \
    --output bs_exp/greedy_output.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en 
