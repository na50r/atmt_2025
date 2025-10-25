import os
import pickle
import logging
import argparse
from seq2seq.data.tokenizer import BPETokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct text data from binary datasets.")
    parser.add_argument("--binary-dir", type=str, required=True,
                        help="Directory containing binary dataset files (e.g., train.en, valid.de).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save reconstructed text files.")
    parser.add_argument("--src-model", type=str, required=True,
                        help="Path to the Source Language SentencePiece/BPE model.")
    parser.add_argument("--tgt-model", type=str, required=True,
                        help="Path to the Target Language SentencePiece/BPE model.")
    parser.add_argument("--source-lang", type=str, required=True,
                        help="Source language code (e.g., en).")
    parser.add_argument("--target-lang", type=str, required=True,
                        help="Target language code (e.g., de).")
    parser.add_argument("--ignore-existing", action="store_true",
                        help="Skip decoding if output file already exists.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress logging output.")
    return parser.parse_args()


def decode_binary_dataset(input_file, output_file, preprocessor: BPETokenizer, ignore_existing=False):
    """Decode binary dataset (list of token IDs) into text using BPETokenizer."""
    if os.path.exists(output_file) and not ignore_existing:
        logging.info(f"File {output_file} already exists, skipping...")
        return

    # Load pickled token sequences
    with open(input_file, "rb") as f:
        tokens_list = pickle.load(f)

    decoded_sentences = []
    for tokens in tokens_list:
        # Convert numpy arrays to lists if needed
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        text = preprocessor.decode(tokens)
        # Optional: remove EOS token if included in the decoded string
        text = text.replace(preprocessor.eos, "").strip()
        decoded_sentences.append(text)

    # Write to output .txt
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(decoded_sentences))

    logging.info(
        f"Decoded {len(decoded_sentences)} sentences -> {output_file}")


def main():
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizers
    src_tokenizer = BPETokenizer(language=args.source_lang)
    tgt_tokenizer = BPETokenizer(language=args.target_lang)

    src_tokenizer.load(args.src_model)
    tgt_tokenizer.load(args.tgt_model)

    # Loop over splits
    for split in ["train", "tiny_train", "valid", "test"]:
        for lang, tok in [(args.source_lang, src_tokenizer), (args.target_lang, tgt_tokenizer)]:
            binary_path = os.path.join(args.binary-dir, f"{split}.{lang}")
            if not os.path.exists(binary_path):
                continue  # skip missing splits
            output_path = os.path.join(args.output_dir, f"{split}.{lang}.txt")
            decode_binary_dataset(binary_path, output_path,
                                  tok, ignore_existing=args.ignore_existing)

    logging.info("Decoding complete!")


if __name__ == "__main__":
    main()
