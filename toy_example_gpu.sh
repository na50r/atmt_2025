#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:15:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=toy_example_gpu.out

# module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# clean up from previous runs
rm -rf toy_example_gpu/data/prepared
rm -rf toy_example_gpu/tokenizers
rm -rf toy_example_gpu/checkpoints
rm -rf toy_example_gpu/logs
rm -f toy_example_gpu/toy_example_output.en

python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ./toy_example_gpu/data/raw \
    --dest-dir ./toy_example_gpu/data/prepared \
    --model-dir ./toy_example_gpu/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 1000 \
    --tgt-vocab-size 1000 \
    --ignore-existing \
    --force-train

python train.py \
    --cuda \
    --data toy_example_gpu/data/prepared/ \
    --src-tokenizer toy_example_gpu/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example_gpu/tokenizers/en-bpe-1000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 10 \
    --log-file toy_example_gpu/logs/train.log \
    --save-dir toy_example_gpu/checkpoints/ \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

python translate.py \
    --cuda \
    --input toy_example_gpu/data/raw/test.cz \
    --src-tokenizer toy_example_gpu/tokenizers/cz-bpe-1000.model \
    --tgt-tokenizer toy_example_gpu/tokenizers/en-bpe-1000.model \
    --checkpoint-path toy_example_gpu/checkpoints/checkpoint_best.pt \
    --batch-size 1 \
    --max-len 100 \
    --output toy_example_gpu/toy_example_output.en \
    --bleu \
    --reference toy_example_gpu/data/raw/test.en
