"""
High-level Translator API for French-English Neural Machine Translation.
Wraps tokenizers, Transformer model, and decoding strategies.
"""

from __future__ import annotations
import torch
from typing import List, Optional

from ..tokenizer.tokenizer import BPETokenizer
from ..tokenizer.vocabulary import BOS_IDX, EOS_IDX, PAD_IDX
from ..model.transformer import Transformer
from ..training.checkpoint import load_checkpoint
from ..inference.greedy_decode import greedy_decode
from ..inference.beam_search import beam_search_decode


class Translator:
    """High-level translation interface."""

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
            num_heads=config.get("num_heads", 4),
            num_encoder_layers=config.get("num_encoder_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            d_ff=config.get("d_ff", 1024),
            dropout=config.get("dropout", 0.1),
            max_len=config.get("max_len", 5000),
            pad_idx=config.get("pad_idx", PAD_IDX),
            tie_weights=config.get("tie_weights", False),
            norm_first=config.get("norm_first", True)
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
        strategy: str = "beam",
        beam_size: int = 4,
        max_len: int = 60
    ) -> str:
        """
        Translate a single French sentence into English.

        Args:
            sentence: Source French text.
            strategy: 'greedy' or 'beam'.
            beam_size: Beam size if strategy is 'beam'.
            max_len: Maximum target tokens to generate.

        Returns:
            Translated English string.
        """
        src_tokens = self.src_tokenizer.encode(sentence, add_bos=True, add_eos=True)

        if strategy.lower() == "greedy":
            out_tokens = greedy_decode(
                model=self.model,
                src_tokens=src_tokens,
                max_len=max_len,
                bos_idx=BOS_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=self.device
            )
        elif strategy.lower() == "beam":
            out_tokens = beam_search_decode(
                model=self.model,
                src_tokens=src_tokens,
                beam_size=beam_size,
                max_len=max_len,
                bos_idx=BOS_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=self.device
            )
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}. Choose 'greedy' or 'beam'.")

        return self.tgt_tokenizer.decode(out_tokens)

    def translate_batch(
        self,
        sentences: List[str],
        strategy: str = "beam",
        beam_size: int = 4,
        max_len: int = 60
    ) -> List[str]:
        """Translate multiple sentences."""
        return [
            self.translate(s, strategy=strategy, beam_size=beam_size, max_len=max_len)
            for s in sentences
        ]
