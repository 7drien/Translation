"""
Training pipeline with Teacher Forcing for Transformer Translation.
Implements the training loop, gradient clipping, evaluation, and checkpointing.
"""

from __future__ import annotations
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any

from ..model.transformer import Transformer
from ..model.masks import create_masks
from ..tokenizer.vocabulary import PAD_IDX
from .validate import evaluate_loss
from .checkpoint import save_checkpoint
from .scheduler import NoamScheduler


def train_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scheduler: Optional[Any],
    device: torch.device,
    clip_grad: float = 1.0,
    pad_idx: int = PAD_IDX
) -> float:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for step, (src, tgt) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing inputs and targets
        decoder_input = tgt[:, :-1]
        expected_output = tgt[:, 1:]

        src_mask, tgt_mask, _ = create_masks(src, decoder_input, pad_idx=pad_idx)

        optimizer.zero_grad()

        logits = model(src, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)

        logits = logits.reshape(-1, logits.size(-1))
        targets = expected_output.reshape(-1)

        loss = criterion(logits, targets)
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        non_pad_tokens = (targets != pad_idx).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


def train_model(
    model: Transformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_epochs: int = 15,
    lr: float = 5e-4,
    warmup_steps: int = 400,
    checkpoint_dir: str = "checkpoints",
    device: Optional[torch.device] = None,
    pad_idx: int = PAD_IDX,
    label_smoothing: float = 0.1,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Full training loop with validation and checkpointing.

    Returns:
        Dictionary with history of 'train_loss' and 'val_loss'.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Label smoothing regularizes predictions and prevents over-confident outputs
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=label_smoothing
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=model.d_model,
        warmup_steps=warmup_steps
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            clip_grad=1.0,
            pad_idx=pad_idx
        )
        history["train_loss"].append(train_loss)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss = evaluate_loss(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                pad_idx=pad_idx
            )
            history["val_loss"].append(val_loss)

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = f"{checkpoint_dir}/best_model.pt"
                save_checkpoint(
                    filepath=best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_loss
                )
                if verbose:
                    print(f"  [Checkpoint] New best model saved to {best_path} (val_loss={val_loss:.4f})")

        elapsed = time.time() - start_time
        if verbose:
            val_str = f" | Val Loss: {val_loss:.4f}" if val_loader is not None else ""
            print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_loss:.4f}{val_str} | Time: {elapsed:.2f}s")

    # Save final model
    final_path = f"{checkpoint_dir}/final_model.pt"
    save_checkpoint(
        filepath=final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=num_epochs,
        val_loss=best_val_loss
    )

    return history
