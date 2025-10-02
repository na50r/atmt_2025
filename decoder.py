import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.decode import decode
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler

import pickle

def decode_to_string(tokenizer, array):
    """
    Takes a tensor of token IDs and decodes it back into a string."""
    if torch.is_tensor(array) and array.dim() == 2:
        return '\n'.join(decode_to_string(tokenizer, t) for t in array)
    return tokenizer.Decode(array.tolist())

def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', required=True, help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument('--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True, help='path to the model file')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str, help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int, help='maximum length of generated sequence')
     
    # Input handling
    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'), weights_only=False)
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    with open(args.input, "rb") as f:
        data = pickle.load(f)
        src_lines_from_array = [decode_to_string(src_tokenizer, torch.tensor(d)).split('\n') for d in data]
        src_lines_from_array = [row[0] for row in src_lines_from_array]
        src_lines = [line.strip() for line in f if line.strip()]
    with open('decoded.txt', 'w') as f:
        for s in src_lines:
            print(s, file=f)

if __name__ == '__main__':
    args = get_args()
    main(args)
