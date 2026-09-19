"""
Byte-Pair Encoding (BPE) training algorithm.
Trains subword vocabularies directly from text corpora.
"""

from __future__ import annotations
import collections
from typing import List, Tuple, Dict, Iterable
from .vocabulary import Vocabulary, SPECIAL_TOKENS
from .tokenizer import BPETokenizer, END_OF_WORD, basic_pre_tokenize, get_pairs_from_word


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 4000,
    min_frequency: int = 2,
    lowercase: bool = True
) -> BPETokenizer:
    """
    Train a BPE tokenizer from an iterable of text strings.

    Args:
        texts: Collection of sentences.
        vocab_size: Target vocabulary size (including special tokens and characters).
        min_frequency: Minimum pair frequency required to perform a merge.
        lowercase: Whether to convert text to lowercase before training.

    Returns:
        Trained BPETokenizer instance.
    """
    # 1. Count word frequencies
    raw_word_counts: Dict[str, int] = collections.defaultdict(int)
    for text in texts:
        if lowercase:
            text = text.lower()
        words = basic_pre_tokenize(text)
        for w in words:
            raw_word_counts[w] += 1

    # Represent each word as a tuple of characters + END_OF_WORD
    # e.g., "chat" -> ('c', 'h', 'a', 't</w>')
    word_freqs: Dict[Tuple[str, ...], int] = {}
    initial_chars = set()

    for word, count in raw_word_counts.items():
        if not word:
            continue
        word_tuple = tuple(list(word) + [END_OF_WORD])
        word_freqs[word_tuple] = count
        for ch in word:
            initial_chars.add(ch)
        initial_chars.add(END_OF_WORD)

    # 2. Initialize vocabulary with special tokens and initial characters
    vocab = Vocabulary(special_tokens=SPECIAL_TOKENS)
    for ch in sorted(initial_chars):
        vocab.add_token(ch)

    merges: List[Tuple[str, str]] = []

    # 3. Iteratively merge most frequent pairs
    while len(vocab) < vocab_size:
        # Count all adjacent pairs
        pair_counts: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        for word_tuple, count in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pair_counts[pair] += count

        if not pair_counts:
            break

        # Find most frequent pair
        best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])

        if best_count < min_frequency:
            break

        merges.append(best_pair)
        merged_token = best_pair[0] + best_pair[1]
        vocab.add_token(merged_token)

        # Apply merge to all words in corpus
        new_word_freqs: Dict[Tuple[str, ...], int] = {}
        first, second = best_pair

        for word_tuple, count in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if (
                    i < len(word_tuple) - 1
                    and word_tuple[i] == first
                    and word_tuple[i + 1] == second
                ):
                    new_word.append(merged_token)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = count

        word_freqs = new_word_freqs

    return BPETokenizer(vocab=vocab, merges=merges, lowercase=lowercase)
