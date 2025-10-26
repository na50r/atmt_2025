#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:0:0
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_assignment3_attempt.out

module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit


# PREPARE DATA
python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ~/shares/cz-en/data/raw \
    --dest-dir ./cz-en-a3-a1/data/prepared \
    --model-dir ./cz-en-a3-a1/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000 \
    --src-model ./cz-en-a3-a1/tokenizers/cz-bpe-8000.model \
    --tgt-model ./cz-en-a3-a1/tokenizers/en-bpe-8000.model

# TRAIN
python train.py \
    --cuda \
    --data cz-en-a3-a1/data/prepared/ \
    --src-tokenizer cz-en-a3-a1/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en-a3-a1/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 7 \
    --log-file cz-en-a3-a1/logs/train.log \
    --save-dir cz-en-a3-a1/checkpoints/ \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 

# TRANSLATE
python translate.py \
    --cuda \
    --input ~/shares/cz-en/data/raw/test.cz \
    --src-tokenizer cz-en-a3-a1/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en-a3-a1/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en-a3-a1/checkpoints/checkpoint_best.pt \
    --output cz-en-a3-a1/output.txt \
    --max-len 300 \
    --bleu \
    --reference ~/shares/cz-en/data/raw/test.en \
    --max-len 300