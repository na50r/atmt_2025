#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=decode.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

python decode_binary_dataset.py \
  --binary-dir ./data-bin \
  --output-dir ./decoded \
  --src-model cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-model cz-en/tokenizers/en-bpe-8000.model \
  --source-lang cz \
  --target-lang en \
  --src-vocab-size 8000 \
  --tgt-vocab-size 8000 