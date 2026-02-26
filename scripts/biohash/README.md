# BioHash - Bio-Inspired Hashing Implementation

Complete implementation of the BioHash algorithm from:

> **"Bio-Inspired Hashing for Unsupervised Similarity Search"**
> Ryali et al., ICML 2020

## 📁 Directory Contents

### 📖 Start Here
- **`00_START_HERE.md`** - Master overview and quick start guide

### 🐍 Core Implementation
- **`biohash_implementation.py`** (18KB) - Base BioHash algorithm for images
- **`biohash_text.py`** (21KB) - Text hashing with TF-IDF/SBERT/Word2Vec
- **`biohash_docuverse.py`** (16KB) - DocUVerse corpus integration

### 🧪 Tests & Examples
- **`validate_biohash.py`** (6KB) - Unit tests for core algorithm
- **`test_biohash_simple.py`** (7KB) - Quick tests on synthetic data
- **`test_text_hashing.py`** (6KB) - Text hashing tests
- **`example_biohash_benchmark.py`** (16KB) - 4 practical examples

### 📚 Documentation
- **`README_BIOHASH.md`** (10KB) - Algorithm documentation
- **`README_TEXT_HASHING.md`** (13KB) - Text hashing guide
- **`BIOHASH_SUMMARY.md`** (14KB) - Deep dive into algorithm
- **`TEXT_HASHING_SUMMARY.md`** (14KB) - Text hashing overview
- **`QUICK_START.md`** (3KB) - Quick reference

### 🖼️ Generated Files
- `biohash_circle_weights.png` - Visualization output

## 🚀 Quick Start

### Option 1: Test Images (MNIST)
```bash
python test_biohash_simple.py
python biohash_implementation.py
```

### Option 2: Test Text Hashing
```bash
python test_text_hashing.py
python biohash_text.py
python example_biohash_benchmark.py
```

## 📖 Documentation Guide

**New to BioHash?**
1. Read `00_START_HERE.md`
2. Run `test_biohash_simple.py`
3. Read `QUICK_START.md`

**Want to hash images?**
1. Read `README_BIOHASH.md`
2. See `BIOHASH_SUMMARY.md` for details

**Want to hash text?**
1. Read `README_TEXT_HASHING.md`
2. See `TEXT_HASHING_SUMMARY.md` for overview
3. Run examples in `example_biohash_benchmark.py`

## 💡 Use Cases

### Image Hashing
- Image similarity search
- Duplicate detection
- Visual clustering

### Text Hashing
- Semantic document search
- Duplicate detection
- Question answering
- Document clustering
- Works with DocUVerse benchmarks

## 🛠️ Installation

```bash
# Minimal (TF-IDF text hashing)
pip install torch numpy scikit-learn

# Full (with Sentence-BERT)
pip install torch numpy scikit-learn sentence-transformers

# For MNIST experiments
pip install torchvision matplotlib
```

## 📊 Performance

**MNIST** (mAP@All %):
- k=16: 54.9% (vs 41.2% ITQ)
- k=32: 55.5% (vs 43.6% ITQ)

**Text** (10K docs):
- Build: 15-35s
- Query: 1-3ms
- Storage: ~128-512 bits/doc

## 🔬 Algorithm Overview

```
Input → Sparse Expansion (m >> d) → Learn W → k-WTA → Binary Hash
```

**Key Innovation**: Learned projections + sparse high-dimensional codes

## 📞 Support

- Check documentation files
- Run test scripts
- See examples in code

---

**Total Files**: 14
**Total Size**: ~370KB
**Paper**: `/home/raduf/sandbox2/docuverse/writeups/2001.04907v2.pdf`
