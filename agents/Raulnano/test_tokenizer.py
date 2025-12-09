"""Lightweight sanity checks for the SentencePiece tokenizer.

Run with: python agents/Raulnano/test_tokenizer.py
This will train/load the tokenizer, perform a few encode/decode checks,
and exit with code 0 on success.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from agents.Raulnano import train


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_encode_decode_round_trip():
    text = "Hello world, this is a tokenizer smoke test."
    ids = train.tokenize(text)
    decoded = train.detokenize(ids)
    assert_true(isinstance(ids, list), "tokenize should return list[int]")
    assert_true(len(ids) > 0, "tokenize should produce at least one id")
    assert_true(len(decoded) > 0, "detokenize should return non-empty string")
    # Basic sanity: decoded should contain alphabetic content
    assert_true(any(c.isalpha() for c in decoded), "decoded text should contain alphabetic content")


def check_special_tokens_present():
    sp = train.tokenizer.sp
    assert_true(sp is not None, "SentencePiece processor should be loaded")
    assert_true(sp.pad_id() == 0, "pad_id should be 0")
    assert_true(sp.bos_id() == 1, "bos_id should be 1")
    assert_true(sp.eos_id() == 2, "eos_id should be 2")
    # Ensure user-defined symbol exists
    pieces = {sp.id_to_piece(i) for i in range(min(sp.get_piece_size(), 32))}
    assert_true("<subtask>" in pieces, "<subtask> should be in vocab")


def check_files_exist():
    model_path = train.TOKENIZER_MODEL_PATH
    vocab_path = train.TOKENIZER_VOCAB_PATH
    assert_true(model_path.exists(), f"Missing tokenizer model at {model_path}")
    assert_true(vocab_path.exists(), f"Missing tokenizer vocab at {vocab_path}")


def main():
    try:
        check_encode_decode_round_trip()
        check_special_tokens_present()
        check_files_exist()
        print("Tokenizer sanity checks passed.")
    except AssertionError as exc:
        print(f"Tokenizer sanity check failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
