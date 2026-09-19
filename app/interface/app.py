"""
Interactive Chatbot CLI for French -> English Translation.
Powered by an optimized Transformer with Beam Search decoding.
"""

from __future__ import annotations
import sys
from typing import Optional
from .api import Translator


def run_chatbot(
    translator: Translator,
    default_beam_size: int = 5
) -> None:
    """Run interactive terminal translation session."""
    print("=" * 60)
    print("🤖  BOT DE TRADUCTION NEURONALE (Français → Anglais)")
    print("    Architecture Transformer avec Beam Search optimisé")
    print("=" * 60)
    print("Commandes disponibles :")
    print("  /beam <taille>  : Ajuster la taille du beam (ex: /beam 5)")
    print("  /aide           : Afficher cette aide")
    print("  /quitter        : Quitter le chatbot")
    print("-" * 60)

    beam_size = default_beam_size

    while True:
        try:
            print(f"\n[Beam Search: k={beam_size}]")
            user_input = input("Français > ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/quitter", "/exit", "/quit", "q"):
                print("Au revoir !")
                break

            if user_input.lower() in ("/aide", "/help"):
                print("Commandes : /beam <taille>, /quitter")
                continue

            if user_input.lower().startswith("/beam") or user_input.lower().startswith("/taille"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    beam_size = max(1, int(parts[1]))
                    print(f"Taille du beam ajustée à : {beam_size}")
                else:
                    print("Usage: /beam <entier> (ex: /beam 5)")
                continue

            # Perform translation using Beam Search
            english_trans = translator.translate(
                sentence=user_input,
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
