"""
High-level Translator API for French-English Neural Machine Translation.
Wraps tokenizers, Transformer model, and Beam Search decoding.
"""

from __future__ import annotations
import torch
from typing import List, Optional

from ..tokenizer.tokenizer import BPETokenizer
from ..tokenizer.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX
from ..model.transformer import Transformer
from ..training.checkpoint import load_checkpoint
from ..inference.beam_search import beam_search_decode


class Translator:
    """High-level translation interface powered by Beam Search."""

    def __init__(
        self,
        model: Transformer,
        src_tokenizer: BPETokenizer,
        tgt_tokenizer: BPETokenizer,
        device: Optional[torch.device] = None
    ) -> None:
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        src_tokenizer_path: str,
        tgt_tokenizer_path: str,
        device: Optional[torch.device] = None
    ) -> Translator:
        """Load a translator directly from saved checkpoint and tokenizer files."""
        src_tokenizer = BPETokenizer.load(src_tokenizer_path)
        tgt_tokenizer = BPETokenizer.load(tgt_tokenizer_path)

        dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=dev)
        config = checkpoint.get("config", {})

        model = Transformer(
            src_vocab_size=config.get("src_vocab_size", len(src_tokenizer.vocab)),
            tgt_vocab_size=config.get("tgt_vocab_size", len(tgt_tokenizer.vocab)),
            d_model=config.get("d_model", 256),
            num_heads=config.get("num_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            d_ff=config.get("d_ff", 1024),
            dropout=config.get("dropout", 0.1),
            max_len=config.get("max_len", 5000),
            pad_idx=config.get("pad_idx", PAD_IDX),
            tie_weights=config.get("tie_weights", True)
        )

        model.load_state_dict(checkpoint["model_state"])
        model.to(dev)
        model.eval()

        return cls(
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=dev
        )

    def translate(
        self,
        sentence: str,
        beam_size: int = 5,
        max_len: int = 60,
        alpha: float = 0.7,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Translate a single French sentence into English using optimized Beam Search.

        Args:
            sentence: Source French text.
            beam_size: Number of hypotheses tracked in beam search (default: 5).
            max_len: Maximum target tokens to generate.
            alpha: Length penalty exponent (default: 0.7).
            no_repeat_ngram_size: Prevent repetitive n-gram loops (default: 3).

        Returns:
            Translated English string.
        """
        src_tokens = self.src_tokenizer.encode(sentence, add_bos=True, add_eos=True)

        out_tokens = beam_search_decode(
            model=self.model,
            src_tokens=src_tokens,
            beam_size=beam_size,
            max_len=max_len,
            bos_idx=BOS_IDX,
            eos_idx=EOS_IDX,
            pad_idx=PAD_IDX,
            alpha=alpha,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=self.device
        )

        return self.tgt_tokenizer.decode(out_tokens)

    def translate_batch(
        self,
        sentences: List[str],
        beam_size: int = 5,
        max_len: int = 60,
        alpha: float = 0.7,
        no_repeat_ngram_size: int = 3
    ) -> List[str]:
        """Translate multiple sentences."""
        return [
            self.translate(
                s,
                beam_size=beam_size,
                max_len=max_len,
                alpha=alpha,
                no_repeat_ngram_size=no_repeat_ngram_size
            )
            for s in sentences
        ]
