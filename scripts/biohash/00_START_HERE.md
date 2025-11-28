# BioHash Implementation - Start Here! 🚀

## What Is This?

Complete implementation of **BioHash** - a bio-inspired hashing algorithm from the paper:

> **"Bio-Inspired Hashing for Unsupervised Similarity Search"**  
> Ryali et al., ICML 2020  
> Paper: `/home/raduf/sandbox2/docuverse/writeups/2001.04907v2.pdf`

## What Can You Do?

### ✅ Hash Images (MNIST)
- Reproduce paper results on MNIST
- Achieve ~55% mAP (vs ~43% for ITQ)
- Fast locality-sensitive hashing

### ✅ Hash Text Documents ⭐ NEW!
- Semantic search across documents
- Duplicate detection
- Document clustering
- Works with DocUVerse corpora

## 📁 File Organization

```
scripts/
│
├── 00_START_HERE.md              ← YOU ARE HERE
│
├── Image Hashing (MNIST)
│   ├── biohash_implementation.py      # Main implementation
│   ├── test_biohash_simple.py         # Quick tests
│   ├── validate_biohash.py            # Unit tests
│   ├── README_BIOHASH.md              # Full documentation
│   ├── BIOHASH_SUMMARY.md             # Algorithm details
│   └── QUICK_START.md                 # Quick reference
│
├── Text Hashing (NEW!)
│   ├── biohash_text.py                # Text hashing module
│   ├── biohash_docuverse.py           # DocUVerse integration
│   ├── example_biohash_benchmark.py   # Practical examples
│   ├── test_text_hashing.py           # Text tests
│   ├── README_TEXT_HASHING.md         # Text hashing docs
│   └── TEXT_HASHING_SUMMARY.md        # Overview
│
└── Paper
    └── ../writeups/2001.04907v2.pdf   # Original paper
```

## 🚀 Quick Start

### Option 1: Test Images (MNIST)

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn

cd /home/raduf/sandbox2/docuverse/scripts

# Run quick tests (1-2 minutes)
python test_biohash_simple.py

# Run full MNIST experiment (15-30 minutes)
python biohash_implementation.py
```

### Option 2: Test Text Hashing ⭐

```bash
# Install dependencies
pip install torch numpy scikit-learn

# Optional: for best quality
pip install sentence-transformers

cd /home/raduf/sandbox2/docuverse/scripts

# Run text tests
python test_text_hashing.py

# Run text demos
python biohash_text.py

# Run practical examples
python example_biohash_benchmark.py
```

## 📖 Documentation

### For Image Hashing
- **Start**: `QUICK_START.md`
- **Full Guide**: `README_BIOHASH.md`
- **Deep Dive**: `BIOHASH_SUMMARY.md`

### For Text Hashing
- **Full Guide**: `README_TEXT_HASHING.md`
- **Overview**: `TEXT_HASHING_SUMMARY.md`

## 💡 Use Cases

### Image Hashing
- ✅ Image similarity search
- ✅ Duplicate image detection
- ✅ Image clustering
- ✅ Visual search engines

### Text Hashing
- ✅ Semantic search
- ✅ Document deduplication
- ✅ Question answering
- ✅ Text clustering
- ✅ Information retrieval

## 🎯 Quick Examples

### Image Hashing

```python
from biohash_implementation import BioHash
import torch

# Train on images
X_train = torch.randn(1000, 784)  # MNIST-like
biohash = BioHash(input_dim=784, num_neurons=320, hash_length=16)
biohash.fit(X_train)

# Generate hash codes
hash_codes = biohash.hash(X_train)
```

### Text Hashing

```python
from biohash_text import BioHashText, TfidfEmbedder

# Documents
docs = ["Machine learning is AI", "Deep learning uses neural nets"]

# Create and train
embedder = TfidfEmbedder()
embedder.fit(docs)

biohash = BioHashText(embedder=embedder, hash_length=16)
biohash.fit(docs)

# Search
results = biohash.search("neural networks", docs, top_k=5)
```

### DocUVerse Integration

```python
from biohash_docuverse import DocUVerseBioHash

# Load corpus
doc_hash = DocUVerseBioHash(embedding_type='tfidf')
doc_hash.load_from_tsv('corpus.tsv')
doc_hash.build_index()

# Search
results = doc_hash.search("your query", top_k=10)

# Save for reuse
doc_hash.save_index('./my_index')
```

## 🎓 How It Works

### The Algorithm

```
1. Input → Sparse Expansion (m >> d)
2. Learn weights W using bio-plausible dynamics
3. k-Winner-Take-All → Binary hash code
4. Similar inputs → Similar hash codes
```

### Key Innovation

**Classical LSH**: Random projections, low-dimensional codes  
**BioHash**: Learned projections, sparse high-dimensional codes

**Result**: Better locality preservation!

## 📊 Performance

### MNIST (mAP@All %)

| Method | k=8 | k=16 | k=32 |
|--------|-----|------|------|
| LSH | 18.1 | 20.3 | 26.2 |
| ITQ | 38.4 | 41.2 | 43.6 |
| **BioHash** | **53.4** | **54.9** | **55.5** |

### Text (10K documents)

| Hash Length | Build Time | Query Time |
|-------------|------------|------------|
| k=16 | 15s | 1.2ms |
| k=32 | 22s | 1.8ms |
| k=64 | 35s | 2.5ms |

## 🔬 Bio-Inspired Features

✅ **Hebbian learning** - "Neurons that fire together, wire together"  
✅ **Competitive learning** - Winner-take-all dynamics  
✅ **Sparse coding** - Only k << m neurons active  
✅ **Local updates** - No backpropagation needed  
✅ **Energy minimization** - Provably decreasing energy

## 🛠️ Requirements

### Minimal (CPU only)
```bash
pip install torch numpy scikit-learn
```

### Full (with GPU & Sentence-BERT)
```bash
pip install torch numpy scikit-learn sentence-transformers
```

## ✅ Validation

Run these to verify everything works:

```bash
# Test images
python validate_biohash.py
python test_biohash_simple.py

# Test text
python test_text_hashing.py
```

## 📚 Learn More

### Papers & Theory
1. Read `README_BIOHASH.md` for algorithm details
2. See paper: `../writeups/2001.04907v2.pdf`
3. Check `BIOHASH_SUMMARY.md` for deep dive

### Practical Usage
1. See `README_TEXT_HASHING.md` for text hashing
2. Run `example_biohash_benchmark.py` for examples
3. Check demos in `biohash_text.py`

## 🎯 Next Steps

1. **Quick test**: `python test_text_hashing.py`
2. **Run demo**: `python biohash_text.py`
3. **Try examples**: `python example_biohash_benchmark.py`
4. **Read docs**: `README_TEXT_HASHING.md`
5. **Use your data**: See examples in docs

## 💡 Tips

- Use **TF-IDF** for speed
- Use **Sentence-BERT** for quality
- Use **GPU** for large corpora
- **Save indexes** for reuse
- Start with **k=16, activity=0.05**

## 📞 Support

- Check documentation files
- Run test scripts
- See examples in code
- Review demos

---

**Happy Hashing!** 🧠🔬🚀
