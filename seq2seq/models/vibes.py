import torch
import torch.nn.functional as F
from queue import PriorityQueue
from itertools import count


class BeamSearch:
    """Beam search for a single sequence."""

    def __init__(self, beam_size, max_len, pad):
        self.beam_size = beam_size
        self.max_len = max_len
        self.pad = pad
        self.nodes = PriorityQueue()  # beams to expand
        self.final = PriorityQueue()  # finished beams
        self._counter = count()

    def add(self, score, node):
        self.nodes.put((score, next(self._counter), node))

    def add_final(self, score, node):
        missing = self.max_len - node.length
        node.sequence = torch.cat(
            (node.sequence.cpu(), torch.tensor([self.pad]*missing).long()))
        self.final.put((score, next(self._counter), node))

    def get_current_beams(self):
        nodes = []
        while not self.nodes.empty() and len(nodes) < self.beam_size:
            node = self.nodes.get()
            nodes.append((node[0], node[2]))
        return nodes

    def get_best(self):
        merged = PriorityQueue()
        for _ in range(self.final.qsize()):
            merged.put(self.final.get())
        for _ in range(self.nodes.qsize()):
            merged.put(self.nodes.get())
        _, _, node = merged.get()
        return node

    def prune(self):
        nodes = PriorityQueue()
        finished = self.final.qsize()
        for _ in range(self.beam_size - finished):
            if not self.nodes.empty():
                nodes.put(self.nodes.get())
        self.nodes = nodes


class BeamSearchNode:
    """Stores a beam node."""

    def __init__(self, sequence, logp, length):
        self.sequence = sequence
        self.logp = logp
        self.length = length

    def eval(self, alpha=0.0):
        normalizer = (5 + self.length)**alpha / (5 + 1)**alpha
        return self.logp / normalizer


def decode_beam_search(model, src_tokens, src_pad_mask, max_out_len,
                       tgt_tokenizer, args, device, beam_size=5, alpha=0.0):
    """Beam search decoding."""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    final_sequences = []

    for i in range(batch_size):
        beam = BeamSearch(beam_size, max_out_len, PAD)
        # root node with BOS
        root = BeamSearchNode(sequence=torch.tensor([BOS], device=device).unsqueeze(0),
                              logp=0.0, length=1)
        beam.add(0.0, root)

        for t in range(max_out_len):
            current_beams = beam.get_current_beams()
            beam.nodes = PriorityQueue()  # reset

            for score, node in current_beams:
                seq = node.sequence.to(device)
                trg_pad_mask = (seq == PAD).unsqueeze(
                    0).unsqueeze(1).unsqueeze(2)
                output = model(
                    src_tokens[i:i+1], src_pad_mask[i:i+1], seq, trg_pad_mask)
                logits = output[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

                top_log_probs, top_tokens = torch.topk(log_probs, beam_size)
                for logp, tok in zip(top_log_probs.tolist(), top_tokens.tolist()):
                    new_seq = torch.cat(
                        [node.sequence, torch.tensor([[tok]], device=device)], dim=1)
                    new_node = BeamSearchNode(sequence=new_seq,
                                              logp=node.logp + logp,
                                              length=node.length + 1)
                    if tok == EOS:
                        beam.add_final(new_node.eval(alpha), new_node)
                    else:
                        beam.add(new_node.eval(alpha), new_node)

            if beam.nodes.empty():
                break

            beam.prune()

        best_node = beam.get_best()
        best_seq = best_node.sequence.squeeze(0).tolist()[1:]  # remove BOS
        if EOS in best_seq:
            idx = best_seq.index(EOS)
            best_seq = best_seq[:idx+1]
        final_sequences.append(best_seq)

    return final_sequences
