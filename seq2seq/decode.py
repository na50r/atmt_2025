import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
import math


def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS,
                           dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    encoder_out = None
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(
            1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        if encoder_out is None:
            encoder_out = model.encoder(src_tokens, src_pad_mask)
        output = model.decoder(encoder_out, src_pad_mask,
                               generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    # Implement relative local threshold pruning, add a third value in beams, the last score
    beams = [(torch.tensor([[BOS]], device=device), 0.0, 0.0)]
    encoder_out = None
    for _ in range(max_out_len):
        new_beams = []
        for seq, score, lscore in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score, lscore))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                if encoder_out is None:
                    encoder_out = model.encoder(src_tokens, src_pad_mask)
                logits = model.decoder(
                    encoder_out, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                last_score = topk_log_probs[:, k].item()
                new_score = score + last_score
                new_beams.append((new_seq, new_score, last_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        # Implementing Relative Local Threshold Pruning
        # Doesn't work well with log probs, use math.exp to get real probs, then scale via rpl value
        curr_best = max(beams, key=lambda x: x[2])[-1]
        updated_beams = []
        for i, b in enumerate(beams):
            print('DEBUG: Beam Cnt', len(beams))
            score = b[-1]
            print('DEBUG: Beam Score', i, score)
            # Relative scaling with log probs not ideal, we use real probs instead
            scaled_score = math.exp(score)
            scaled_best = math.exp(curr_best)
            if scaled_score <= (scaled_best * 0.02):
                print(f'DEBUG: Remove {scaled_score:.2f} <= {scaled_best} * RPL (RPL=0.02)')
            else:
                updated_beams.append(b)
        beams = sorted(updated_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(seq[0, -1].item() == EOS for seq, _, _ in beams):
            break
    best_seq, _, _  = beams[0]
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    return [best_seq.squeeze(0).tolist()]