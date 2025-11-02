import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel
import torch.nn.functional as F

def beam_search_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device, beam_size=5):
    # Thise function was created by ChatGPT but then further commented/assessed by the student.
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Only works for batch size 1 (for now)"

    # The reasoon why ChatGPT suggested separating encoder_out/decoder_out rather than using the model directly is because the student pointed out that the model essentially just uses them directly
    # Thus ChatGPT realized it can safe computation because encoder_out will alaways be the same for the same src_tokens
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
            # This is the same output as in the original decode function, just obtained through model.decoder rather than model's forward pass
            logits = model.decoder(
                encoder_out, 
                src_pad_mask, 
                seq, 
                trg_pad_mask)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            
            # Instead of using argmax on logits, ChatGPT took log_probabilities
            # This is important because for beam search, the best target sequence is chosen via cumulative scores. 
            # Logits are negative and not normalized, not nice for cumulation.
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            # ChatGPT choose topK but this is really just beam_size. We don't choose randomly but the best k log_probs
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
            
            # For each log_prob, we create a new sequence
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
    return [tokens]


def beam_search_decode_v2(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device, beam_size=5):
    # Modified by ChatGPT to be more optimized
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()

    batch_size = src_tokens.size(0)
    assert batch_size == 1, "Only batch size 1 supported."

    encoder_out = model.encoder(src_tokens, src_pad_mask)

    beam = [(torch.tensor([[BOS]], device=device), 0.0)]
    completed = []

    for step in range(max_out_len):
        # Separate active and completed beams
        active = [(s, sc) for s, sc in beam if s[0, -1] != EOS]
        completed.extend([(s, sc) for s, sc in beam if s[0, -1] == EOS])

        # If no active beams remain, stop early
        if not active:
            break

        # Stack sequences for one decoder forward pass
        beam_seqs = torch.cat([s for s, _ in active], dim=0)
        trg_pad_mask = (beam_seqs == PAD).unsqueeze(1).unsqueeze(2)

        logits = model.decoder(
            encoder_out.repeat(len(active), 1, 1),
            src_pad_mask.repeat(len(active), 1, 1),
            beam_seqs,
            trg_pad_mask
        )

        log_probs = F.log_softmax(
            logits[:, -1, :], dim=-1)  # (beam_active, vocab)

        all_candidates = []
        for i, (_, score) in enumerate(active):
            topk_log_probs, topk_idx = log_probs[i].topk(beam_size)
            for k in range(beam_size):
                next_token = topk_idx[k].view(1, 1)
                new_seq = torch.cat([beam_seqs[i:i+1], next_token], dim=1)
                all_candidates.append(
                    (new_seq, score + topk_log_probs[k].item()))

        # Prune to top beam_size total (active + completed)
        all_candidates.extend(completed)
        all_candidates = sorted(
            all_candidates, key=lambda x: x[1], reverse=True)
        beam = all_candidates[:beam_size]

        # Optional: stop early if we have enough completed beams
        if len(completed) >= beam_size:
            break

    # Pick the best completed sequence
    if completed:
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_seq, _ = completed[0]
    else:
        best_seq, _ = beam[0]

    tokens = best_seq[0, 1:].tolist()
    if EOS in tokens:
        tokens = tokens[:tokens.index(EOS)+1]
    return [tokens]

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
