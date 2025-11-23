#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=a5_test_beam_3_alpha.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

OUTDIR=cz-en-a1-base
OUTDIR2=alphas
rm -rf $OUTDIR2
mkdir $OUTDIR2
SEED=512
mkdir $OUTDIR
cat ~/shares/cz-en/data/raw/test.cz | head -n 100 > test.cz
cat ~/shares/cz-en/data/raw/test.en | head -n 100 > test.en

# TRANSLATE
for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python translate.py \
        --cuda \
        --input test.cz \
        --src-tokenizer $OUTDIR/tokenizers/cz-bpe-8000.model \
        --tgt-tokenizer $OUTDIR/tokenizers/en-bpe-8000.model \
        --checkpoint-path $OUTDIR/checkpoints/checkpoint_best.pt \
        --output $OUTDIR2/output_beam_3_alpha_$i.txt \
        --max-len 300 \
        --bleu \
        --reference test.en \
        --seed $SEED \
        --beam-size 3 \
        --alpha=$i