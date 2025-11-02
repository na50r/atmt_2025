import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
import torch.nn.functional as F


def beam_search(prediction, k=10):
    # Source: https://stackoverflow.com/a/76661466
    batch_size, seq_length, vocab_size = prediction.shape
    log_prob, indices = prediction[:, 0, :].topk(k, sorted=True)
    indices = indices.unsqueeze(-1).to(prediction.device)  # ensure same device
    for n1 in range(1, seq_length):
        log_prob_temp = log_prob.unsqueeze(-1) + \
            prediction[:, n1, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index_temp = log_prob_temp.view(
            batch_size, -1).topk(k, sorted=True)
        # move to same device
        idx_begin = (index_temp // vocab_size).to(indices.device)
        idx_concat = index_temp % vocab_size

        new_indices = torch.zeros(
            (batch_size, k, n1+1), dtype=torch.int64, device=prediction.device)
        for n2 in range(batch_size):
            new_indices[n2, :, :-1] = indices[n2][idx_begin[n2]]
            new_indices[n2, :, -1] = idx_concat[n2]
        indices = new_indices
    return indices, log_prob


def beam_search_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device, beam_size=5):
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Only works for batch size 1 (for now)"

    # Encoder output (same for all beam hypotheses)
    encoder_out = model.encoder(src_tokens, src_pad_mask)

    # Initialize beam with BOS token
    # (sequence, log_prob)
    beam = [(torch.tensor([[BOS]], device=device), 0.0)]
    completed = []

    for _ in range(max_out_len):
        candidates = []
        for seq, score in beam:
            if seq[0, -1].item() == EOS:
                completed.append((seq, score))
                continue

            trg_pad_mask = (seq == PAD).unsqueeze(1).unsqueeze(2)
            logits = model.decoder(
                encoder_out, src_pad_mask, seq, trg_pad_mask)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            log_probs = F.log_softmax(next_token_logits, dim=-1)

            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
            for k in range(beam_size):
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, k].item()
                candidates.append((new_seq, new_score))

        # Select top beam_size sequences by cumulative log prob
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        beam = candidates[:beam_size]

        if len(completed) >= beam_size:
            break

    # If we have completed sequences, pick the best one
    if completed:
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq, _ = completed[0]
    else:
        best_seq, _ = beam[0]

    # Remove BOS and trim after EOS
    tokens = best_seq[0, 1:].tolist()
    if EOS in tokens:
        tokens = tokens[:tokens.index(EOS)+1]
    return tokens


def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        # batch_size, seq_len, vocab_size
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        #print("DEBUG output", output.shape)
        # Get the logits for the last time step
        # batch_size, vocab_size
        next_token_logits = output[:, -1, :]  # last time step (actually, last token in sequence)
        #print("DEBUG next_token_logits", next_token_logits.shape)
        # batch_size, 1 (ID of token with highest logit)
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy
        3
        #print("DEBUG next_tokens", next_tokens.shape)

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
