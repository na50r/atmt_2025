from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler
from seq2seq import models, utils
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq.decode import decode
import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from seq2seq.beam import BeamSearch, BeamSearchNode
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


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
    parser.add_argument('--seed', default=42, type=int,
                        help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', required=True,
                        help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument(
        '--src-tokenizer', help='path to source sentencepiece tokenizer', required=True)
    parser.add_argument(
        '--tgt-tokenizer', help='path to target sentencepiece tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True,
                        help='path to the model file')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str,
                        help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int,
                        help='maximum length of generated sequence')

    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true',
                        help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str,
                        help='Path to the reference file (one sentence per line, required if --bleu is set)')

    return parser.parse_args()


def main(args):
    """ Main translation function' """
    # Load arguments from checkpoint
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s,
                            l: default_restore_location(s, 'cpu'), weights_only=False)
    args_loaded = argparse.Namespace(
        **{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)
    # make_batch = utils.make_batch_input(device='cuda' if args.cuda else 'cpu',
    #                                     pad=src_tokenizer.pad_id(),
    #                                     max_seq_len=args.max_len)

    # batch input sentences

    def batch_iter(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]

    # Build model and criterion
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    logging.info('Loaded a model from checkpoint {:s}'.format(
        args.checkpoint_path))

    # Read input sentences
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]

    # Encode input sentences
    src_encoded = [torch.tensor(src_tokenizer.Encode(
        line, out_type=int, add_eos=True)) for line in src_lines]
    # trim to max_len
    max_seq_len = min(model.encoder.pos_embed.size(1), args.max_len)
    # src_encoded = [s[:max_seq_len] for s in src_encoded]
    src_encoded = [s if len(s) <= max_seq_len else s[:max_seq_len]
                   for s in src_encoded]

    DEVICE = 'cuda' if args.cuda else 'cpu'
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    UNK = tgt_tokenizer.unk_id()
    print(f'PAD ID: {PAD}, BOS ID: {BOS}, EOS ID: {EOS}\n\
          PAD token: "{src_tokenizer.IdToPiece(PAD)}", BOS token: "{tgt_tokenizer.IdToPiece(BOS)}", EOS token: "{tgt_tokenizer.IdToPiece(EOS)}"')

    # Clear output file
    if args.output is not None:
        with open(args.output, 'w', encoding="utf-8") as out_file:
            out_file.write('')

    def postprocess_ids(ids, pad, bos, eos):
        """Remove leading BOS, truncate at first EOS, remove PADs."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # remove leading BOS if present
        if len(ids) > 0 and ids[0] == bos:
            ids = ids[1:]
        # truncate at EOS (do not include EOS)
        if eos in ids:
            ids = ids[:ids.index(eos)]
        # remove PAD tokens (typically trailing, but remove any)
        ids = [i for i in ids if i != pad]
        return ids

    def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids):
        """Convert token ids to a detokenized string using the target tokenizer."""
        ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
        # Use tokenizer.Decode to produce properly detokenized text
        return tokenizer.Decode(ids)

    translations = []
    start_time = time.perf_counter()

    make_batch = utils.make_batch_input(
        device=DEVICE, pad=src_tokenizer.pad_id(), max_seq_len=args.max_len)

    progress_bar = tqdm(batch_iter(src_encoded, args.batch_size))
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()

    # Iterate over the test set
    all_hyps = {}
    for i, sample in enumerate(progress_bar):

        # Create a beam search object or every input sentence in batch
        batch_size = sample['src_tokens'].shape[0]
        searches = [BeamSearch(
            args.beam_size, args.max_len - 1, UNK) for i in range(batch_size)]

        with torch.no_grad():
            # Compute the encoder output
            encoder_out = model.encoder(
                sample['src_tokens'], sample['src_lengths'])
            # __QUESTION 1: What is "go_slice" used for and what do its dimensions represent?
            go_slice = \
                torch.ones(sample['src_tokens'].shape[0], 1).fill_(
                    EOS).type_as(sample['src_tokens'])
            if args.cuda:
                go_slice = utils.move_to_cuda(go_slice)

            # import pdb;pdb.set_trace()

            # Compute the decoder output at the first time step
            decoder_out, _ = model.decoder(go_slice, encoder_out)

            # __QUESTION 2: Why do we keep one top candidate more than the beam size?
            log_probs, next_candidates = torch.topk(torch.log(torch.softmax(decoder_out, dim=2)),
                                                    args.beam_size+1, dim=-1)

        #  Create number of beam_size beam search nodes for every input sentence
        for i in range(batch_size):
            for j in range(args.beam_size):
                best_candidate = next_candidates[i, :, j]
                backoff_candidate = next_candidates[i, :, j+1]
                best_log_p = log_probs[i, :, j]
                backoff_log_p = log_probs[i, :, j+1]
                next_word = torch.where(
                    best_candidate == UNK, backoff_candidate, best_candidate)
                log_p = torch.where(
                    best_candidate == UNK, backoff_log_p, best_log_p)
                log_p = log_p[-1]

                # Store the encoder_out information for the current input sentence and beam
                emb = encoder_out['src_embeddings'][:, i, :]
                lstm_out = encoder_out['src_out'][0][:, i, :]
                final_hidden = encoder_out['src_out'][1][:, i, :]
                final_cell = encoder_out['src_out'][2][:, i, :]
                try:
                    mask = encoder_out['src_mask'][i, :]
                except TypeError:
                    mask = None

                node = BeamSearchNode(searches[i], emb, lstm_out, final_hidden, final_cell,
                                      mask, torch.cat((go_slice[i], next_word)), log_p, 1)
                # __QUESTION 3: Why do we add the node with a negative score?
                searches[i].add(-node.eval(args.alpha), node)

        # import pdb;pdb.set_trace()
        # Start generating further tokens until max sentence length reached
        for _ in range(args.max_len-1):

            # Get the current nodes to expand
            nodes = [n[1] for s in searches for n in s.get_current_beams()]
            if nodes == []:
                break  # All beams ended in EOS

            # Reconstruct prev_words, encoder_out from current beam search nodes
            prev_words = torch.stack([node.sequence for node in nodes])
            encoder_out["src_embeddings"] = torch.stack(
                [node.emb for node in nodes], dim=1)
            lstm_out = torch.stack([node.lstm_out for node in nodes], dim=1)
            final_hidden = torch.stack(
                [node.final_hidden for node in nodes], dim=1)
            final_cell = torch.stack(
                [node.final_cell for node in nodes], dim=1)
            encoder_out["src_out"] = (lstm_out, final_hidden, final_cell)
            try:
                encoder_out["src_mask"] = torch.stack(
                    [node.mask for node in nodes], dim=0)
            except TypeError:
                encoder_out["src_mask"] = None

            with torch.no_grad():
                # Compute the decoder output by feeding it the decoded sentence prefix
                decoder_out, _ = model.decoder(prev_words, encoder_out)

            # see __QUESTION 2
            log_probs, next_candidates = torch.topk(
                torch.log(torch.softmax(decoder_out, dim=2)), args.beam_size+1, dim=-1)

            #  Create number of beam_size next nodes for every current node
            for i in range(log_probs.shape[0]):
                for j in range(args.beam_size):

                    best_candidate = next_candidates[i, :, j]
                    backoff_candidate = next_candidates[i, :, j+1]
                    best_log_p = log_probs[i, :, j]
                    backoff_log_p = log_probs[i, :, j+1]
                    next_word = torch.where(
                        best_candidate == UNK, backoff_candidate, best_candidate)
                    log_p = torch.where(
                        best_candidate == UNK, backoff_log_p, best_log_p)
                    log_p = log_p[-1]
                    next_word = torch.cat((prev_words[i][1:], next_word[-1:]))

                    # Get parent node and beam search object for corresponding sentence
                    node = nodes[i]
                    search = node.search

                    # __QUESTION 4: How are "add" and "add_final" different?
                    # What would happen if we did not make this distinction?

                    # Store the node as final if EOS is generated
                    if next_word[-1] == EOS:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                   next_word)), node.logp, node.length
                        )
                        search.add_final(-node.eval(args.alpha), node)

                    # Add the node to current nodes for next iteration
                    else:
                        node = BeamSearchNode(
                            search, node.emb, node.lstm_out, node.final_hidden,
                            node.final_cell, node.mask, torch.cat((prev_words[i][0].view([1]),
                                                                   next_word)), node.logp + log_p, node.length + 1
                        )
                        search.add(-node.eval(args.alpha), node)

            # #import pdb;pdb.set_trace()
            # __QUESTION 5: What happens internally when we prune our beams?
            # How do we know we always maintain the best sequences?
            for search in searches:
                search.prune()

        # Segment into sentences
        best_sents = torch.stack(
            [search.get_best()[1].sequence[1:].cpu() for search in searches])
        decoded_batch = best_sents.numpy()
        # import pdb;pdb.set_trace()

        output_sentences = [decoded_batch[row, :]
                            for row in range(decoded_batch.shape[0])]

        # __QUESTION 6: What is the purpose of this for loop?
        temp = list()
        for sent in output_sentences:
            first_eos = np.where(sent == EOS)[0]
            if len(first_eos) > 0:
                temp.append(sent[:first_eos[0]])
            else:
                temp.append(sent)
        output_sentences = temp

        # Convert arrays of indices into strings of words
        output_sentences = [decode_sentence(
            tgt_tokenizer, sent) for sent in output_sentences]
        output_sentences = [postprocess_ids(sent) for sent in output_sentences]

        for ii, sent in enumerate(output_sentences):
            all_hyps[int(sample['id'].data[ii])] = sent

    # Write to file
    if args.output is not None:
        with open(args.output, 'w') as out_file:
            for sent_id in range(len(all_hyps.keys())):
                out_file.write(all_hyps[sent_id] + '\n')


if __name__ == '__main__':
    args = get_args()
    main(args)
