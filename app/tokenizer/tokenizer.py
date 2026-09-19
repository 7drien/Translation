"""
Byte-Pair Encoding (BPE) Tokenizer built from scratch.
Implements subword segmentation, encoding into token IDs, and decoding back to text.
"""

from __future__ import annotations
import json
import re
from typing import List, Tuple, Dict, Optional, Set
from .vocabulary import (
    Vocabulary, PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN,
    PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, SPECIAL_TOKENS
)

END_OF_WORD = "</w>"


def basic_pre_tokenize(text: str) -> List[str]:
    """
    Split text into words and punctuation marks while preserving accents.
    Handles French and English punctuation nicely.
    """
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Separate punctuation from words
    pattern = r"[\w]+|[^\w\s]"
    tokens = re.findall(pattern, text, flags=re.UNICODE)
    return tokens


def get_pairs_from_word(word_tokens: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Return set of adjacent symbol pairs in a word representation."""
    pairs = set()
    for i in range(len(word_tokens) - 1):
        pairs.add((word_tokens[i], word_tokens[i + 1]))
    return pairs


class BPETokenizer:
    """
    Custom Byte-Pair Encoding (BPE) Tokenizer.
    Trained without any external dependencies.
    """

    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        lowercase: bool = True
    ) -> None:
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.merges: List[Tuple[str, str]] = merges if merges is not None else []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        self.lowercase = lowercase
        # Cache for segmenting known words
        self.cache: Dict[str, List[str]] = {}

    def _bpe_segment_word(self, word: str) -> List[str]:
        """Segment a single word into subword tokens using learned BPE merges."""
        if word in self.cache:
            return self.cache[word]

        if not word:
            return []

        # Represent word as tuple of individual characters with end-of-word marker
        word_tuple = tuple(list(word) + [END_OF_WORD])
        pairs = get_pairs_from_word(word_tuple)

        if not pairs:
            # Single character word with </w>
            self.cache[word] = [word + END_OF_WORD]
            return self.cache[word]

        while True:
            # Find the pair that occurred earliest in learned merges (min rank)
            min_pair = None
            min_rank = float("inf")
            for pair in pairs:
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            # If no pair in this word is in our merge table, we are done
            if min_pair is None or min_rank == float("inf"):
                break

            # Apply merge to the word
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1

            word_tuple = tuple(new_word)
            if len(word_tuple) == 1:
                break
            pairs = get_pairs_from_word(word_tuple)

        result = list(word_tuple)
        self.cache[word] = result
        return result

    def tokenize(self, text: str) -> List[str]:
        """Convert a sentence into a list of BPE subword tokens."""
        if self.lowercase:
            text = text.lower()

        words = basic_pre_tokenize(text)
        subwords: List[str] = []
        for word in words:
            segmented = self._bpe_segment_word(word)
            subwords.extend(segmented)
        return subwords

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        ids = [self.vocab.token_to_index(tok) for tok in tokens]

        if add_bos:
            ids = [BOS_IDX] + ids
        if add_eos:
            ids = ids + [EOS_IDX]

        return ids

    def decode(self, token_ids: List[int], strip_special: bool = True) -> str:
        """Decode a list of token IDs back into a human-readable string."""
        tokens = []
        for idx in token_ids:
            if strip_special and idx in (PAD_IDX, BOS_IDX, EOS_IDX):
                continue
            tokens.append(self.vocab.index_to_token(idx))

        # Reconstruct text from subwords with </w>
        words = []
        current_word = ""

        for token in tokens:
            if token.endswith(END_OF_WORD):
                current_word += token[:-len(END_OF_WORD)]
                words.append(current_word)
                current_word = ""
            else:
                current_word += token

        if current_word:
            words.append(current_word)

        text = " ".join(words)

        # Fix spacing around punctuation: e.g. "student ." -> "student."
        text = re.sub(r"\s+([.,!?;:'\"])", r"\1", text)
        text = re.sub(r"([(\[{])\s+", r"\1", text)
        text = re.sub(r"\s+([)\]}])", r"\1", text)
        # French apostrophe fix: e.g. "l ' élève" -> "l'élève"
        text = re.sub(r"(\b[a-zA-ZÀ-ÿ])\s*'\s*", r"\1'", text)

        return text.strip()

    def to_dict(self) -> dict:
        """Serialize tokenizer state to dictionary."""
        return {
            "vocab": self.vocab.to_dict(),
            "merges": [list(pair) for pair in self.merges],
            "lowercase": self.lowercase
        }

    @classmethod
    def from_dict(cls, data: dict) -> BPETokenizer:
        """Instantiate tokenizer from dictionary."""
        vocab = Vocabulary.from_dict(data["vocab"])
        merges = [tuple(pair) for pair in data["merges"]]
        lowercase = data.get("lowercase", True)
        return cls(vocab=vocab, merges=merges, lowercase=lowercase)

    def save(self, filepath: str) -> None:
        """Save tokenizer configuration to JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> BPETokenizer:
        """Load tokenizer configuration from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
