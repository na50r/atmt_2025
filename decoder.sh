#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_decoder.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# DECODE
python decoder.py \
    --cuda \
    --input cz-en/data/prepared/test.cz \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/output.txt \
    --max-len 300
