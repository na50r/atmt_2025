#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=exp_big.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# TRANSLATE
python translate_v2.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en-a3-a1/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en-a3-a1/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en-a3-a1/checkpoints/checkpoint_best.pt \
    --output exp/output_big.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \