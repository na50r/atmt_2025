import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel


def beam_search(prediction, k=10):
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

def decode_beam_search(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_width: int = 5):
    """Decodes a sequence using beam search."""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    # Step 1: Prepare initial input (BOS token)
    generated = torch.full((batch_size, 1), BOS,
                           dtype=torch.long, device=device)

    # Step 2: Get logits for all positions (simulate what your model would output)
    # We'll build a "prediction tensor" like (batch_size, seq_length, vocab_size)
    # by running the model autoregressively for max_out_len steps
    predictions = []
    for t in range(max_out_len):
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        trg_pad_mask = (generated == PAD).unsqueeze(
            1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        output = model(src_tokens, src_pad_mask,
                       generated, trg_pad_mask).to(device)
        # (batch_size, 1, vocab_size)
        predictions.append(output[:, -1, :].unsqueeze(1))
        # For next step, just append argmax token to simulate next input
        generated = torch.cat(
            [generated, output[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)

    # Concatenate predictions to shape: (batch_size, max_out_len, vocab_size)
    prediction_tensor = torch.cat(predictions, dim=1)

    # Step 3: Run beam search
    topk_indices, topk_log_probs = beam_search(prediction_tensor, k=beam_width)

    # Step 4: Post-process sequences (remove BOS, cut at EOS)
    predicted_tokens = []
    for batch_seq in topk_indices:  # batch_seq shape: (beam_width, seq_length)
        # take the top-1 beam for simplicity, or you could return all beams
        seq = batch_seq[0].tolist()  # top-1 beam
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)

    return predicted_tokens


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
        print("DEBUG output", output.shape)
        # Get the logits for the last time step
        # batch_size, vocab_size
        next_token_logits = output[:, -1, :]  # last time step
        print("DEBUG next_token_logits", next_token_logits.shape)
        # batch_size, 1 (ID of token with highest logit)
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy
        print("DEBUG next_tokens", next_tokens.shape)

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
