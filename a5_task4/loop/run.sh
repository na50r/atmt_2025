#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=loop_%x.out   # output file for each job

# $1 = threshold
# $2 = beam_size

THRESHOLD=$1
BEAM_SIZE=$2

module load miniforge3
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

MODEL_DIR=~/shares/groups/turing/a5/cz-en-a5-base
OUT_DIR=~/shares/groups/turing/a5/a5_task4/loop
mkdir -p $OUT_DIR
SCRIPTS=~/shares/groups/turing/a5
SEED=512

# Prepare small test set (only done once, no harm if repeated)
head -n 100 ~/shares/cz-en/data/raw/test.cz > test.cz
head -n 100 ~/shares/cz-en/data/raw/test.en > test.en

python $SCRIPTS/translate.py \
    --cuda \
    --input test.cz \
    --src-tokenizer $MODEL_DIR/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer $MODEL_DIR/tokenizers/en-bpe-8000.model \
    --checkpoint-path $MODEL_DIR/checkpoints/checkpoint_best.pt \
    --output "$OUT_DIR/a5_task4_beam${BEAM_SIZE}_threshold${THRESHOLD}.txt" \
    --max-len 300 \
    --bleu \
    --reference test.en \
    --seed $SEED \
    --beam-size $BEAM_SIZE \
    --es-threshold $THRESHOLD

