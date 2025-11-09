# Vibe Coding
In this document, I go over how I vibe coded Beam Search. Initially it was in the report but because report had to be 2-3 pages, I did not want to go over the prompt, the code and my analysis of it. The purpose of this is to prove that I did not just copy-paste but spent quite some time understanding how the code implemented Greedy search and then was able to prompt ChatGPT to provide me with a working Beam search implementation. Since vibe coding is officially allowed for these assignments, this document is just for me to prove that I did not just copy-paste. 

## Prompt
```
Can you explain decoding in transformer based Machine Translation?

In my code, I see the following 
>A model that has an encoder and decoder; both transformer based. 
>The model takes src, src_mask, tgt; tgt_pad_mask as input. 
>src and src_mask are passed into the encoder 
>The encoder output, tgt and tgt_mask are passed into the decoder.

Understanding: tgt is a sequence that is generated in a loop until the EOS of token is reached. So we start with a BOS token, pass in src and that into the model and get back the next token. Append that to tgt and continue until EOS is reached.

<CODE of decode function, seq2seq/decode.py>

This decode function implement greedy decoding with the model. I ask, how would I be able to implement Beam search with this? Because we know that the decoder model outputs logits and we select the next token by using argmax on all logits. But for beam search, we presumably have to select topk first. Then do we pass in the different variants of those top k into the model? Wouldnt that take longer?
```
As you can see, it's not just asking ChatGPT to refactor the decode function to implement beam search. I actually tried myself a bunch of time and failed. I understood how decode function essentially calls the deocder repeatedly with the same SRC sequence everytime. The only thing that changes is the generated token sequence. So the first input is source sequence + BOS token, the second input is source sequence + BOS token + first generated token and so on. Because the next token is selected with argmax, it's straight forward and fast. When I looked into beam search, I found a bunch of articles and images explaining how we explore multiple possibilities. We choose not just the best but k of the best and then choose the best sequence using cumulative scores rather than logits.

### Insight in Code
`>The encoder output, tgt and tgt_mask are passed into the decoder.`

I assume this one is somewhat relevant. In the original code, we have:
```
# Forward pass: use only the generated tokens so far
output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
```
The forward pass is:
```
def forward(self, src, src_mask, trg, trg_pad_mask):
    return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)
```
In inference, `src_tokens` is the same all the time, so we can optimize by doing:
```
encoder_out = model.encoder(src_tokens, src_pad_mask)
output = model.decoder(encoder_out, src_pad_mask, generated, trg_pad_mask)
```
Which saves computation since the `src_tokens` don't have to be re-encoded anymore. For training, encoding makes sense since we want the model to learn. For inference, the model should be deterministic, so `encoder_out` remains unchanged and can be reused.

## Response
* Omitting ChatGPT long-winded whatever since I ignored it and mainly worked with the code it provided.
```python
import torch
import torch.nn.functional as F

def beam_search_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, device, beam_size=5):
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    
    batch_size = src_tokens.size(0)
    assert batch_size == 1, "For simplicity, start with batch_size=1"

    # Encoder output (same for all beam hypotheses)
    encoder_out = model.encoder(src_tokens, src_pad_mask)

    # Initialize beam with BOS token
    beam = [(torch.tensor([[BOS]], device=device), 0.0)]  # (sequence, log_prob)
    completed = []

    for _ in range(max_out_len):
        candidates = []
        for seq, score in beam:
            if seq[0, -1].item() == EOS:
                completed.append((seq, score))
                continue

            trg_pad_mask = (seq == PAD).unsqueeze(1).unsqueeze(2)
            logits = model.decoder(encoder_out, src_pad_mask, seq, trg_pad_mask)
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
```
The only thing in this code that failed was `return tokens` which had to be changed to `return [tokens]`. This was because later on, the tokens, i.e., ids, where turned back into text sequence by sequence. So this decode function shouldn't return the best sequence but a list of sequences with length 1. This will not work if batch_size>1, so cannot be used for training.

## Analysis of Response
What Beam search essentially does is choosing the target token sequence with the highest cumulative log probability. Whereas greedy decoding in the original code chose the token with highest logit value at each iteration of the outermost loop (range of `max_out_len`). The softmax is used to convert logits to log probabilities for cumulation. (Logits are often negative, so you cannot cumulate logits directly). It uses Pytorch `topk` method to choose the best $k$ next tokens rather than the best one ($k$=`beam_size`). The `candidates` list stores all sequences and their cumulative scores. By sorting that list by those scores in reverse (largest score first) and then choosing the first `beam_size` candidates for the beam, we basically expand in the direction of higher cumulative scores. The code terminates the loop if completed is not empty or if reached the `max_out_len`. An obvious issue with this algorithm is that decoding will take longer because we are exploring more target sequence alternatives and call the decoder multiple times. Whereas for greedy decoding, the decoder is always called once for the next token. So at a `beam_size=5` we may expect a 5x slower translation speed. Optimizations are possible by vectorizing for-loops with tensor operations but I could not find time to do that and that was not asked by the task, so I left this to the tutor (if they choose to use this implementation for the next assignment).