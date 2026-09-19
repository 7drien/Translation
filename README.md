# Système de Traduction Neuronale (Français → Anglais)
### Architecture Transformer Encodeur-Décodeur codée *from scratch* (sans modèles pré-entraînés)

Ce projet implémente un système complet de traduction automatique neuronale (NMT) du français vers l'anglais conformément aux spécifications du fichier `agents.md`.

Aucun modèle pré-entraîné (Hugging Face, MarianMT, etc.) ni poids existant n'est utilisé. Toute la logique du modèle, de la tokenisation et du décodage a été implémentée de zéro en Python avec PyTorch (utilisé uniquement pour les calculs tensoriels, l'autograd et la gestion GPU/CPU).

---

## 🏛️ Architecture du Projet

```text
Translation/
├── app/
│   ├── tokenizer/               # Tokenizer BPE codé from scratch
│   │   ├── vocabulary.py        # Gestion du vocabulaire et tokens spéciaux (<PAD>, <UNK>, <BOS>, <EOS>)
│   │   ├── tokenizer.py         # Segmentation sous-mots BPE, encodage et décodage
│   │   └── train_bpe.py         # Algorithme d'apprentissage des fusions BPE
│   │
│   ├── data/                    # Préparation & prétraitement des données
│   │   ├── download.py          # Génération du corpus spécialisé (tourisme, hôtellerie, transport, quotidien)
│   │   ├── clean.py             # Nettoyage, normalisation unicode, filtrage par longueur & ratio
│   │   ├── split.py             # Découpage train (80%), valid (10%), test (10%)
│   │   └── dataset.py           # TranslationDataset et collation dynamique avec padding
│   │
│   ├── model/                   # Architecture Transformer Seq2Seq from scratch
│   │   ├── embeddings.py        # TokenEmbedding & PositionalEncoding sinusoïdal
│   │   ├── attention.py         # MultiHeadAttention (Self-Attention & Cross-Attention)
│   │   ├── masks.py             # Masques de padding et causal (look-ahead)
│   │   ├── encoder.py           # EncoderLayer & TransformerEncoder (Pre-LN)
│   │   ├── decoder.py           # DecoderLayer & TransformerDecoder (Pre-LN)
│   │   └── transformer.py       # Modèle Seq2Seq complet reliant encodeur et décodeur
│   │
│   ├── training/                # Pipeline d'entraînement
│   │   ├── train.py             # Boucle d'entraînement avec Teacher Forcing & Label Smoothing
│   │   ├── validate.py          # Évaluation de la perte sur le jeu de validation
│   │   ├── scheduler.py         # Scheduler d'apprentissage Noam (warmup + decay)
│   │   └── checkpoint.py        # Sauvegarde et chargement des checkpoints
│   │
│   ├── inference/               # Stratégies de décodage autoregressif
│   │   ├── greedy_decode.py     # Décodage glouton (argmax pas à pas)
│   │   └── beam_search.py       # Décodage Beam Search avec pénalité de longueur
│   │
│   ├── evaluation/              # Métriques & analyse
│   │   ├── chrf.py              # Métrique chrF (F-score n-grammes de caractères)
│   │   └── error_analysis.py    # Détection de répétitions, omissions, hallucinations
│   │
│   └── interface/               # Couche Chatbot & API
│       ├── api.py               # Classe Translator haut-niveau
│       └── app.py               # Interface chatbot interactive en ligne de commande
│
├── main.py                      # Pipeline complet automatisé (étapes 1 à 13)
└── agents.md                    # Cahier des charges et spécifications initiales
```

---

## 🚀 Utilisation Rapide

### 1. Lancer le pipeline complet (Entraînement + Évaluation)

```bash
.venv/bin/python main.py --epochs 15
```

Ce script exécute successivement :
1. La génération et le nettoyage du dataset bilingue.
2. L'entraînement des tokenizers BPE français et anglais.
3. Le **test de mémorisation** (étape 6 de `agents.md` : vérification que le modèle mémorise parfaitement 50 phrases pour prouver la validité mathématique des masques et du calcul de gradient).
4. L'entraînement complet avec validation et sauvegarde du meilleur modèle dans `checkpoints/best_model.pt`.
5. La comparaison **Greedy vs Beam Search**.
6. Le calcul des métriques et le rapport d'analyse d'erreurs.

### 2. Lancer le Chatbot de Traduction Interactif

```bash
.venv/bin/python main.py --chat
```
Ou directement via le module d'interface :
```bash
.venv/bin/python app/interface/app.py checkpoints/best_model.pt data/tokenizers/tokenizer_fr.json data/tokenizers/tokenizer_en.json
```

Dans le chatbot :
- Tapez votre phrase en français (ex: `Je voudrais réserver une chambre pour deux personnes.`).
- Tapez `/mode greedy` ou `/mode beam` pour changer la méthode de décodage à la volée.
- Tapez `/beam 5` pour modifier la taille du beam search.
- Tapez `/quitter` pour sortir.

---

## 🔬 Détails des Composants

### 1. Tokenizer BPE
- Représentation de chaque mot sous forme de caractères initiaux + `</w>`.
- Comptage itératif des paires de symboles les plus fréquentes.
- Fusion et constitution progressive d'un vocabulaire de sous-mots.
- Gestion robuste des tokens spéciaux : `<PAD>` (0), `<UNK>` (1), `<BOS>` (2), `<EOS>` (3).

### 2. Attention Multi-Tête & Masques
- Implémentation matricielle sans `nn.MultiheadAttention` :
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
- **Masque de padding** : empêche les têtes d'attention de se focaliser sur les tokens `<PAD>`.
- **Masque causal** : triangulaire supérieur, interdit au décodeur de regarder les tokens futurs.

### 3. Inférence & Décodage
- **Greedy Decoding** : Sélectionne à chaque étape le token de probabilité maximale.
- **Beam Search** : Maintient les $K$ meilleures hypothèses avec normalisation par la longueur :
  $$\text{score}(y) = \frac{\log P(y)}{\left(\frac{5 + |y|}{6}\right)^\alpha}$$
