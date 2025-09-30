#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:15:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=toy_example.out

# module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python translate.py \
    --input toy_example/data/raw/test.cz \
    --src-tokenizer toy_example/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example/checkpoints/checkpoint_best.pt \
    --batch-size 1 \
    --max-len 100 \
    --output toy_example/exp_toy_example_outpu.en \
    --bleu \
    --reference toy_example/data/raw/test.en \
    --cuda
