import sacrebleu
with open('cz-en/reference.txt', 'r', encoding='utf-8') as f:
    ref = [line.strip() for line in f if line.strip()]
    print(len(ref))

with open('cz-en/output.txt', 'r') as f:
    hyp = [line.strip() for line in f if line.strip()]
    print(len(ref))
bleu = sacrebleu.corpus_bleu(hyp, [ref])
print(bleu.format())