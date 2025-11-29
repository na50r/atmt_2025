#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=a5_task3_beam3.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

MODEL_DIR=~/shares/groups/turing/a5/cz-en-a5-base
OUT_DIR=~/shares/groups/turing/a5/cz-en-a5-base/a5_task3
SCRIPTS=~/shares/groups/turing/a5

SEED=512
cat ~/shares/cz-en/data/raw/test.cz | head -n 100 > test.cz
cat ~/shares/cz-en/data/raw/test.en | head -n 100 > test.en

# TRANSLATE
python $SCRIPTS/translate.py \
    --cuda \
    --input test.cz \
    --src-tokenizer $MODEL_DIR/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer $MODEL_DIR/tokenizers/en-bpe-8000.model \
    --checkpoint-path $MODEL_DIR/checkpoints/checkpoint_best.pt \
    --output $OUT_DIR/a5_task3_beam3.txt \
    --max-len 300 \
    --bleu \
    --reference test.en \
    --seed $SEED \
    --beam-size 3