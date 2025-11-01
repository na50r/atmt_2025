import torch
import torch.nn.functional as F
from itertools import count
from queue import PriorityQueue
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel


class BeamSearch(object):
    """ Defines a beam search object for a single input sentence. """
    def __init__(self, beam_size, max_len, pad):

        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad

        self.nodes = PriorityQueue() # beams to be expanded
        self.final = PriorityQueue() # beams that ended in EOS

        self._counter = count() # for correct ordering of nodes with same score

    def add(self, score, node):
        """ Adds a new beam search node to the queue of current nodes """
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        """ Adds a beam search path that ended in EOS (= finished sentence) """
        # ensure all node paths have the same length for batch ops
        missing = self.max_len - node.length
        node.sequence = torch.cat((node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        """ Returns beam_size current nodes with the lowest negative log probability """
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        """ Returns final node with the lowest negative log probability """
        # Merge EOS paths and those that were stopped by
        # max sequence length (still in nodes)
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            node = self.final.get()
            merged.put(node)

        for _ in range(self.nodes.qsize()):
            node = self.nodes.get()
            merged.put(node)

        node = merged.get()
        node = (node[0], node[2])

        return node

    def prune(self):
        """ Removes all nodes but the beam_size best ones (lowest neg log prob) """
        nodes = PriorityQueue()
        # Keep track of how many search paths are already finished (EOS)
        finished = self.final.qsize()
        for _ in range(self.beam_size-finished):
            node = self.nodes.get()
            nodes.put(node)
        self.nodes = nodes


class BeamSearchNode(object):
    """ Defines a search node and stores values important for computation of beam search path"""
    def __init__(self, search, emb, lstm_out, final_hidden, final_cell, mask, sequence, logProb, length):

        # Attributes needed for computation of decoder states
        self.sequence = sequence
        self.emb = emb
        self.lstm_out = lstm_out
        self.final_hidden = final_hidden
        self.final_cell = final_cell
        self.mask = mask

        # Attributes needed for computation of sequence score
        self.logp = logProb
        self.length = length

        self.search = search

    def eval(self, alpha=0.0):
        """ Returns score of sequence up to this node 

        params: 
            :alpha float (default=0.0): hyperparameter for
            length normalization described in in
            https://arxiv.org/pdf/1609.08144.pdf (equation
            14 as lp), default setting of 0.0 has no effect
        
        """
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer


def decode_beam_search(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor,
                       max_out_len: int, tgt_tokenizer, beam_size: int, device: torch.device,
                       alpha=0.0):
    """
    Decodes sequences using beam search.
    """
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    final_sequences = []

    for i in range(batch_size):
        # Create BeamSearch object for each batch item
        beam = BeamSearch(beam_size, max_out_len, PAD)
        # Initialize root node with BOS token
        root = BeamSearchNode(
            search=beam,
            emb=None,  # placeholder if needed for your model
            lstm_out=None,
            final_hidden=None,
            final_cell=None,
            mask=None,
            sequence=torch.tensor([BOS], device=device).unsqueeze(
                0),  # shape (1, 1)
            logProb=0.0,
            length=1
        )
        beam.add(0.0, root)

        for t in range(max_out_len):
            # Get current beams
            current_beams = beam.get_current_beams()
            beam.nodes = PriorityQueue()  # reset nodes for next expansion

            for score, node in current_beams:
                seq = node.sequence.to(device)
                # Prepare trg_pad_mask
                trg_pad_mask = (seq == PAD).unsqueeze(
                    0).unsqueeze(1).unsqueeze(2)
                # Forward pass
                output = model(
                    src_tokens[i:i+1],  # single example
                    src_pad_mask[i:i+1],
                    seq,
                    trg_pad_mask
                )
                logits = output[:, -1, :]  # shape (1, vocab_size)
                log_probs = F.log_softmax(
                    logits, dim=-1).squeeze(0)  # shape (vocab_size,)

                # Expand top k candidates
                top_log_probs, top_tokens = torch.topk(log_probs, beam_size)
                for logp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    new_seq = torch.cat(
                        [node.sequence, torch.tensor([[tok]], device=device)], dim=1)
                    new_node = BeamSearchNode(
                        search=beam,
                        emb=None,
                        lstm_out=None,
                        final_hidden=None,
                        final_cell=None,
                        mask=None,
                        sequence=new_seq,
                        logProb=node.logp + logp,
                        length=node.length + 1
                    )
                    if tok == EOS:
                        beam.add_final(new_node.eval(alpha), new_node)
                    else:
                        beam.add(new_node.eval(alpha), new_node)

            if beam.nodes.empty():
                break  # all beams finished

            beam.prune()  # keep top beam_size nodes

        # Get best sequence for this batch element
        _, best_node = beam.get_best()
        # Remove initial BOS
        best_seq = best_node.sequence.squeeze(0).tolist()[1:]
        # Trim after EOS
        if EOS in best_seq:
            idx = best_seq.index(EOS)
            best_seq = best_seq[:idx+1]
        final_sequences.append(best_seq)

    return final_sequences