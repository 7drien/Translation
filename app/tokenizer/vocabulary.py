"""
Vocabulary management for French-English Translation.
Handles special tokens, mappings between tokens and integer IDs, and serialization.
"""

from __future__ import annotations
import json
from typing import Dict, List, Optional, Iterable

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class Vocabulary:
    """Bi-directional mapping between tokens (strings) and numerical IDs."""

    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS
        for token in tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """Add a token if not present, returns its ID."""
        if token not in self.token_to_id:
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = token
            self.token_to_id[token] = new_id
            return new_id
        return self.token_to_id[token]

    def add_tokens(self, tokens: Iterable[str]) -> None:
        """Add multiple tokens to the vocabulary."""
        for token in tokens:
            self.add_token(token)

    def token_to_index(self, token: str) -> int:
        """Get ID of token, returning UNK_IDX if unknown."""
        return self.token_to_id.get(token, UNK_IDX)

    def index_to_token(self, idx: int) -> str:
        """Get token string from ID, returning UNK_TOKEN if not found."""
        return self.id_to_token.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        return len(self.id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __getitem__(self, item: str | int) -> str | int:
        if isinstance(item, str):
            return self.token_to_index(item)
        elif isinstance(item, int):
            return self.index_to_token(item)
        raise TypeError(f"Invalid item type: {type(item)}")

    def to_dict(self) -> dict:
        """Serialize vocabulary to dictionary."""
        return {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> Vocabulary:
        """Instantiate vocabulary from dictionary."""
        vocab = cls(special_tokens=[])
        vocab.token_to_id = data["token_to_id"]
        vocab.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return vocab

    def save(self, filepath: str) -> None:
        """Save vocabulary to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> Vocabulary:
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
