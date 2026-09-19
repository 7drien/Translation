"""
Interactive Chatbot CLI for French -> English Translation.
Allows interactive conversation and translation with customizable decoding strategies.
"""

from __future__ import annotations
import sys
from typing import Optional
from .api import Translator


def run_chatbot(
    translator: Translator,
    default_strategy: str = "beam",
    default_beam_size: int = 4
) -> None:
    """Run interactive terminal session."""
    print("=" * 60)
    print("🤖  BOT DE TRADUCTION NEURONALE (Français → Anglais)")
    print("    Architecture Transformer entraînée from scratch")
    print("=" * 60)
    print("Commandes disponibles :")
    print("  /mode [greedy|beam]  : Basculer la stratégie de décodage")
    print("  /beam <taille>       : Ajuster la taille du beam (ex: /beam 5)")
    print("  /aide                : Afficher cette aide")
    print("  /quitter             : Quitter le chatbot")
    print("-" * 60)

    strategy = default_strategy
    beam_size = default_beam_size

    while True:
        try:
            print(f"\n[Mode: {strategy.upper()}{f' (taille={beam_size})' if strategy == 'beam' else ''}]")
            user_input = input("Français > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quitter", "/exit", "/quit", "q"):
                print("Au revoir !")
                break

            if user_input.lower() in ("/aide", "/help"):
                print("Commandes : /mode [greedy|beam], /beam <n>, /quitter")
                continue

            if user_input.lower().startswith("/mode"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].lower() in ("greedy", "beam"):
                    strategy = parts[1].lower()
                    print(f"Stratégie modifiée : {strategy}")
                else:
                    print("Usage: /mode greedy  ou  /mode beam")
                continue

            if user_input.lower().startswith("/beam"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    beam_size = max(1, int(parts[1]))
                    strategy = "beam"
                    print(f"Taille du beam ajustée à : {beam_size}")
                else:
                    print("Usage: /beam <entier>")
                continue

            # Perform translation
            english_trans = translator.translate(
                sentence=user_input,
                strategy=strategy,
                beam_size=beam_size
            )

            print(f"Anglais  > {english_trans}")

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrompue. Au revoir !")
            break
        except Exception as e:
            print(f"Erreur lors de la traduction : {e}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python app/interface/app.py <checkpoint.pt> <src_tok.json> <tgt_tok.json>")
        sys.exit(1)

    t = Translator.from_checkpoint(
        checkpoint_path=sys.argv[1],
        src_tokenizer_path=sys.argv[2],
        tgt_tokenizer_path=sys.argv[3]
    )
    run_chatbot(t)
