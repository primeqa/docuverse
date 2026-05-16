#!/usr/bin/env python3
"""
English Bias Diagnostic for embedding models using FLiP
(Factorized Linear Projection).

Inspired by:
  "FLiP: Towards understanding and interpreting multimodal multilingual
   sentence embeddings" (arXiv:2604.18109v1, Kesiraju et al. 2026)

Method:
  Train a factorized linear probe W = A@B that maps sentence embeddings
  to bag-of-words logits. If a model is language-agnostic, a probe trained
  on English embeddings should transfer well to other languages (same content
  → similar embedding → same keywords recoverable). Large transfer gap
  signals English bias.

Usage:
  python scripts/english_bias_diagnostic.py
  python scripts/english_bias_diagnostic.py --model path/to/model --rank 128
"""

import argparse
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "it", "its", "he", "she", "they", "we", "i", "you", "my", "your",
    "his", "her", "their", "our", "this", "that", "these", "those",
    "with", "from", "by", "as", "if", "then", "than", "when", "where",
    "who", "which", "what", "how", "all", "any", "each", "every", "no",
    "so", "up", "out", "about", "into", "through", "during", "before",
    "after", "above", "below", "between", "more", "also", "just",
}


def build_vocab(texts: list[str], max_vocab: int = 5000, min_freq: int = 2) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        counts.update(t for t in tokens if t not in STOPWORDS)
    words = [w for w, c in counts.most_common(max_vocab) if c >= min_freq]
    return {w: i for i, w in enumerate(words)}


def text_to_bow(text: str, vocab: dict[str, int]) -> np.ndarray:
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    vec = np.zeros(len(vocab), dtype=np.float32)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    return vec


# ---------------------------------------------------------------------------
# FLiP model  W = A @ B
# ---------------------------------------------------------------------------

class FLiP(nn.Module):
    """Factorized Linear Projection: embed_dim -> vocab logits via low-rank W=A@B."""

    def __init__(self, embed_dim: int, vocab_size: int, rank: int = 256):
        super().__init__()
        self.B = nn.Linear(embed_dim, rank, bias=False)   # rank × dim
        self.A = nn.Linear(rank, vocab_size, bias=False)  # vocab × rank
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias + self.A(self.B(x))


def train_flip(
    embeddings: np.ndarray,
    bows: np.ndarray,
    vocab_size: int,
    embed_dim: int,
    rank: int = 256,
    epochs: int = 40,
    lr: float = 5e-3,
    batch_size: int = 256,
    l1_lambda: float = 1e-4,
    device: str = "cpu",
    verbose: bool = False,
) -> FLiP:
    model = FLiP(embed_dim, vocab_size, rank).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5, min_lr=1e-5)

    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    Y = torch.tensor(bows, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
    log_softmax = nn.LogSoftmax(dim=-1)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            row_sums = yb.sum(dim=1, keepdim=True).clamp(min=1e-9)
            yb_prob = yb / row_sums

            logits = model(xb)
            ce_loss = -(yb_prob * log_softmax(logits)).sum(dim=-1).mean()
            l1_loss = l1_lambda * model.A.weight.abs().mean()
            loss = ce_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        scheduler.step(epoch_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}")

        # early stopping
        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                break

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def keyword_accuracy(
    model: FLiP,
    embeddings: np.ndarray,
    ref_texts: list[str],
    vocab: dict[str, int],
    device: str = "cpu",
) -> float:
    """Fraction of reference keywords recovered in top-k, averaged over sentences."""
    idx2word = {i: w for w, i in vocab.items()}
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    accs = []
    with torch.no_grad():
        logits = model(X)
        for i, text in enumerate(ref_texts):
            ref_tokens = set(re.findall(r'\b[a-z]{3,}\b', text.lower())) & set(vocab)
            if not ref_tokens:
                continue
            k = min(len(ref_tokens), len(vocab))
            top_idx = torch.topk(logits[i], k).indices.cpu().tolist()
            predicted = {idx2word[j] for j in top_idx}
            accs.append(len(predicted & ref_tokens) / len(ref_tokens))
    return float(np.mean(accs)) * 100.0 if accs else 0.0


def jaccard_index(
    model_a: FLiP,
    model_b: FLiP,
    embeddings: np.ndarray,
    vocab: dict[str, int],
    device: str = "cpu",
    k: int = 10,
) -> float:
    """Jaccard overlap between top-k keywords from two probes on the same embeddings."""
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_a = model_a(X)
        logits_b = model_b(X)
    jaccards = []
    for i in range(len(embeddings)):
        sa = set(torch.topk(logits_a[i], k).indices.cpu().tolist())
        sb = set(torch.topk(logits_b[i], k).indices.cpu().tolist())
        inter = len(sa & sb)
        union = len(sa | sb)
        jaccards.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jaccards)) * 100.0


def anisotropy_score(embeddings: np.ndarray) -> float:
    """Average cosine similarity between random pairs; high = anisotropic (concentrated)."""
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(embeddings), size=(500, 2))
    sims = [
        float(np.dot(embeddings[i], embeddings[j]) /
              (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9))
        for i, j in idx if i != j
    ]
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

LANGUAGE_PAIRS = {
    # opus_config: (lang_code, lang_name, linguistic_distance)
    "de-en": ("de", "German",   "close"),
    "en-fr": ("fr", "French",   "close"),
    "en-es": ("es", "Spanish",  "close"),
    "en-hi": ("hi", "Hindi",    "distant"),
    "en-ja": ("ja", "Japanese", "very distant"),
    "en-zh": ("zh", "Chinese",  "very distant"),
}


def load_parallel(opus_config: str, lang_code: str, n_train: int, n_test: int):
    """Load parallel EN-XX sentences from OPUS-100 test split."""
    from datasets import load_dataset  # lazy import

    ds = load_dataset("Helsinki-NLP/opus-100", opus_config, split="test")
    en_sents, xx_sents = [], []
    for row in ds:
        t = row["translation"]
        en_sents.append(t["en"])
        xx_sents.append(t[lang_code])

    total = len(en_sents)
    n_train = min(n_train, total - n_test)
    tr = slice(0, n_train)
    te = slice(n_train, n_train + n_test)
    return en_sents[tr], xx_sents[tr], en_sents[te], xx_sents[te]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FLiP-based English Bias Diagnostic for Embedding Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="/home/raduf/sandbox2/docuverse/ibm-granite/granite-embedding-97m-multilingual-r2",
        help="SentenceTransformer model path or HuggingFace model ID",
    )
    parser.add_argument("--rank", type=int, default=128, help="FLiP factorization rank")
    parser.add_argument("--vocab_size", type=int, default=3000, help="Max vocabulary size")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--n_train", type=int, default=1500, help="Training sentences per language pair")
    parser.add_argument("--n_test", type=int, default=400, help="Test sentences per language pair")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu", help="torch device (cpu / cuda)")
    parser.add_argument("--verbose", action="store_true", help="Show training loss")
    args = parser.parse_args()

    device = args.device

    print()
    print("=" * 65)
    print("  FLiP English Bias Diagnostic  (arXiv:2604.18109v1)")
    print("=" * 65)
    print(f"  Model      : {Path(args.model).name}")
    print(f"  Rank       : {args.rank}   Vocab: {args.vocab_size}   Epochs: {args.epochs}")
    print()

    # Load embedding model
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    encoder = SentenceTransformer(args.model)
    embed_dim = encoder.get_embedding_dimension()
    print(f"  dim={embed_dim}")
    print()

    # Accumulate results across language pairs
    bias_scores = {}  # lang_name -> acc drop
    lang_results = {}

    shared_vocab = None  # built from first language pair's EN training text

    for opus_cfg, (lang_code, lang_name, distance) in LANGUAGE_PAIRS.items():
        print(f"── {lang_name} ({distance}) ──────────────────────────────")

        # Load parallel data
        en_tr, xx_tr, en_te, xx_te = load_parallel(
            opus_cfg, lang_code, args.n_train, args.n_test
        )
        print(f"  Data: {len(en_tr)} train / {len(en_te)} test sentences")

        # Build shared English vocabulary from first pair's EN training text
        if shared_vocab is None:
            shared_vocab = build_vocab(en_tr, max_vocab=args.vocab_size)
            print(f"  Vocabulary: {len(shared_vocab)} words (built from EN training)")

        vocab = shared_vocab

        # Encode
        print("  Encoding...", end=" ", flush=True)
        en_embs_tr = encoder.encode(en_tr, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        xx_embs_tr = encoder.encode(xx_tr, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        en_embs_te = encoder.encode(en_te, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        xx_embs_te = encoder.encode(xx_te, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
        print("done")

        # Build BoW targets (always EN reference text)
        en_bows_tr = np.array([text_to_bow(t, vocab) for t in en_tr])

        # Train FLiP on EN embeddings
        print("  Training FLiP_EN (English embeddings → EN vocabulary)...", end=" ", flush=True)
        flip_en = train_flip(
            en_embs_tr, en_bows_tr, len(vocab), embed_dim,
            rank=args.rank, epochs=args.epochs, batch_size=args.batch_size,
            device=device, verbose=args.verbose,
        )
        print("done")

        # Train FLiP_XX on XX embeddings (same EN BoW targets)
        print(f"  Training FLiP_{lang_code.upper()} ({lang_name} embeddings → EN vocabulary)...", end=" ", flush=True)
        flip_xx = train_flip(
            xx_embs_tr, en_bows_tr, len(vocab), embed_dim,
            rank=args.rank, epochs=args.epochs, batch_size=args.batch_size,
            device=device, verbose=args.verbose,
        )
        print("done")

        # Evaluate on test split (EN reference text for accuracy)
        acc_en_on_en = keyword_accuracy(flip_en, en_embs_te, en_te, vocab, device)
        acc_en_on_xx = keyword_accuracy(flip_en, xx_embs_te, en_te, vocab, device)
        acc_xx_on_xx = keyword_accuracy(flip_xx, xx_embs_te, en_te, vocab, device)
        jacc = jaccard_index(flip_en, flip_xx, xx_embs_te, vocab, device)

        # Anisotropy
        aniso_en = anisotropy_score(en_embs_te)
        aniso_xx = anisotropy_score(xx_embs_te)

        bias_drop = acc_en_on_en - acc_en_on_xx  # gap: EN probe on EN vs EN probe on XX

        lang_results[lang_name] = {
            "distance": distance,
            "acc_en_on_en": acc_en_on_en,
            "acc_en_on_xx": acc_en_on_xx,
            "acc_xx_on_xx": acc_xx_on_xx,
            "jaccard": jacc,
            "aniso_en": aniso_en,
            "aniso_xx": aniso_xx,
            "bias_drop": bias_drop,
        }
        bias_scores[lang_name] = bias_drop

        print(f"  FLiP_EN  → EN test:  {acc_en_on_en:5.1f}%")
        print(f"  FLiP_EN  → {lang_code.upper()} test:  {acc_en_on_xx:5.1f}%  (EN probe on {lang_name} embeddings)")
        print(f"  FLiP_{lang_code.upper()} → {lang_code.upper()} test:  {acc_xx_on_xx:5.1f}%  ({lang_name} probe on {lang_name} embeddings)")
        print(f"  Jaccard (EN↔{lang_code.upper()} probes on {lang_code.upper()} emb): {jacc:5.1f}%")
        print(f"  Anisotropy  EN={aniso_en:.3f}  {lang_code.upper()}={aniso_xx:.3f}")
        print()

    # ── Final summary ──────────────────────────────────────────────────────
    print("=" * 65)
    print("  ENGLISH BIAS SUMMARY")
    print("=" * 65)
    print(f"  {'Language':<12}  {'Distance':<13}  {'EN→EN':>6}  {'EN→XX':>6}  {'XX→XX':>6}  {'Drop':>6}  {'Jaccard':>8}")
    print(f"  {'-'*12}  {'-'*13}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")

    all_drops = []
    for lang, r in lang_results.items():
        print(
            f"  {lang:<12}  {r['distance']:<13}  "
            f"{r['acc_en_on_en']:5.1f}%  {r['acc_en_on_xx']:5.1f}%  "
            f"{r['acc_xx_on_xx']:5.1f}%  {r['bias_drop']:+5.1f}pp  "
            f"{r['jaccard']:6.1f}%"
        )
        all_drops.append(r["bias_drop"])

    mean_drop = float(np.mean(all_drops))
    print()
    print(f"  Mean EN→XX accuracy drop : {mean_drop:+.1f}pp")
    print()

    # Verdict using paper's key finding: large drop = English bias
    print("  Interpretation (FLiP paper framework):")
    print("  ┌─────────────────────────────────────────────────────────┐")
    if mean_drop < 5:
        verdict = "LOW BIAS — embeddings are well language-aligned"
        interp = (
            "  │ The EN-trained probe transfers well to all languages.    │\n"
            "  │ Semantic concepts are linearly encoded language-         │\n"
            "  │ agnostically. Model is suitable for multilingual use.   │"
        )
    elif mean_drop < 15:
        verdict = "MODERATE BIAS — mild English preference"
        interp = (
            "  │ EN probe transfers reasonably but with degradation.     │\n"
            "  │ Distant languages (Hindi, CJK) may underperform.       │\n"
            "  │ Consider multilingual fine-tuning for better coverage.  │"
        )
    else:
        verdict = "HIGH BIAS — strong English-centric representations"
        interp = (
            "  │ EN probe fails on non-English embeddings. The model     │\n"
            "  │ clusters semantics around English lexical structure.    │\n"
            "  │ Cross-lingual retrieval will degrade significantly for  │\n"
            "  │ linguistically distant languages.                       │"
        )

    print(f"  │ Verdict: {verdict:<50} │")
    print(interp)
    print("  └─────────────────────────────────────────────────────────┘")

    # Per-group summary
    close_drops = [r["bias_drop"] for l, r in lang_results.items() if r["distance"] == "close"]
    distant_drops = [r["bias_drop"] for l, r in lang_results.items() if r["distance"] != "close"]
    print()
    print(f"  Close languages avg drop   : {np.mean(close_drops):+.1f}pp")
    print(f"  Distant languages avg drop : {np.mean(distant_drops):+.1f}pp")
    print()
    print("  Key: EN→EN = EN probe accuracy on EN embeddings (baseline)")
    print("       EN→XX = EN probe accuracy on XX embeddings (transfer)")
    print("       XX→XX = XX probe accuracy on XX embeddings (ceiling)")
    print("       Drop  = EN→EN minus EN→XX (English bias signal)")
    print("       Jaccard = keyword overlap between EN and XX probes")
    print("=" * 65)


if __name__ == "__main__":
    main()
