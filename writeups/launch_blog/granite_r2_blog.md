# Granite Embedding R2: Setting New Standards for Enterprise Retrieval

*IBM Research AI introduces next-generation embedding models that don't compromise between speed and accuracy*

---

When it comes to enterprise information retrieval, organizations face a persistent challenge: existing embedding models force you to choose between accuracy and speed, between long-context support and commercial licensing, between general-purpose performance and domain-specific excellence.

Today, we're introducing the Granite Embedding R2 models—a comprehensive family of retrieval models designed to eliminate these tradeoffs.

## What's New in R2?

The Granite Embedding R2 release includes three models, all available under Apache 2.0 license:

- **granite-embedding-english-r2** (149M parameters): Our flagship model with 768-dimensional embeddings
- **granite-embedding-small-english-r2** (47M parameters): A first-of-its-kind efficient model with 384-dimensional embeddings
- **granite-embedding-reranker-english-r2** (149M parameters): A cross-encoder for precision ranking

These models deliver three critical improvements over our first-generation release:

1. **16x expanded context length** from 512 to 8,192 tokens—meeting modern document processing requirements
2. **19-44% faster inference** than comparable models, without sacrificing accuracy
3. **State-of-the-art performance** across text, code, long-documents, conversational queries, and tabular data

*(Too impatient to read about training pipelines and benchmarks? We get it—skip straight to the [code examples](#getting-started-with-code-examples) and start embedding things.)*

## Built on Modern Foundations

The R2 models leverage the ModernBERT architecture, incorporating recent advances in encoder design:

- Alternating attention mechanisms for efficiency
- Rotary positional embeddings enabling flexible context lengths
- Flash Attention support for optimized inference

We trained these models on 2 trillion tokens from high-quality sources including GneissWeb, Wikipedia, and Granite Code data. Every dataset underwent comprehensive governance review, with screening for personal information and profanity—because enterprise deployments demand transparency and responsible AI practices.

## A Novel Training Pipeline

What sets Granite R2 apart is our five-stage training methodology:

**1. Retrieval-Oriented Pretraining**: Using RetroMAE to train rich [CLS] representations without explicit contrastive objectives

**2. Tabular Pretraining**: A breakthrough approach for handling structured data. Traditional embedding models struggle with tables containing numerical data and limited context. Our solution? We generated synthetic summaries for 8 million tables using Mistral-7B, then modified the RetroMAE objective to predict masked tokens over summaries rather than table content itself. This forces the encoder to align table structure with natural language descriptions.

**3. Contrastive Finetuning**: Training on large-scale semi-supervised pairs with improved contrastive loss

**4. Contrastive Distillation**: Rather than simply finetuning on hard negatives, we distill knowledge from a Mistral-7B teacher model trained on high-quality triples. This approach yields larger performance gains than traditional hard-negative training.

**5. Domain Adaptation**: Specialized training for multi-turn conversational retrieval

This pipeline enables a single model family to excel across remarkably diverse tasks.

## Performance That Speaks for Itself

We evaluated Granite R2 on six open source retrieval benchmarks part of MTEB benchmarks (MTEB v2, CoIR, TableIR, LongEmbed, MTRAG, and MLDR), and the results demonstrate clear leadership in both accuracy and speed.

### Accuracy: State-of-the-Art Across the Board

![Average Retrieval Performance across 6 Benchmarks](R2AveragePerformance.png)

As the chart shows, **granite-embedding-english-r2 achieves the highest average performance at 59.5 NDCG@10**, outperforming all comparable open-source models - including models that are twice its size. Even our efficient **granite-embedding-small-english-r2 scores 55.6**, surpassing many larger open-source competitors.

For a comprehensive analysis of results, please refer to our [research paper](https://arxiv.org/abs/2508.21085).

### Speed: Industry-Leading Efficiency

![Encoding Speed Comparison](R2SpeedComparison.png)

Performance benchmarks often overlook a critical real-world constraint: encoding speed. When you're ingesting millions of documents with frequent updates, speed directly impacts operational costs and user experience.

We benchmarked encoding speed using 23,000 IBM technical documents (averaging 6,393 characters, ranging from 10 to 475,001 characters, details in the [research paper](https://arxiv.org/abs/2508.21085)):

- **granite-english-r2**: 144 documents/second
- **granite-small-r2**: 199 documents/second (fastest in its class)

These speeds represent **19-44% improvements over leading competitors**, despite the R2 models having slightly more parameters than R1. The ModernBERT architecture's optimizations—particularly Flash Attention—enable this efficiency gain.

The speed advantage becomes even more pronounced with the small model, which processes nearly 200 documents per second while maintaining competitive accuracy. This makes it ideal for real-time applications and high-throughput ingestion pipelines.

## Complete Retrieval Ecosystem

The reranker model completes the retrieval pipeline. Built on granite-embedding-english-r2, it uses a PListMLE loss objective for position-aware ranking:

- BEIR: 55.4 (vs. 53.1 for retriever alone)
- MLDR: 44.4 (vs. 41.6 for retriever alone)

This retrieve-and-rerank framework maximizes both recall and precision without severe computational overhead.

## Enterprise-Ready from Day One

Every Granite model prioritizes enterprise requirements:

**Data Governance**: Comprehensive clearance process capturing content description, intended use, data classification, licensing information, usage restrictions, and personal information assessment

**Licensing**: Apache 2.0—no restrictions on commercial use, no proprietary training data limitations

**Transparency**: Fully documented training data sources, architectural decisions, and evaluation methodology

## Why This Matters

Information retrieval isn't just about finding documents—it's about enabling AI systems to access relevant knowledge efficiently. Whether you're building RAG applications, semantic search engines, or recommendation systems, embedding quality and speed determine what's possible.

Granite R2 models don't force you to choose between accuracy and speed, between long-context support and efficiency, between general-purpose capability and domain-specific performance. They deliver all of it.

In an era where milliseconds matter and accuracy cannot be compromised, Granite R2 doesn't just meet the standard—it sets it.

## Getting Started with Code Examples

Using Granite Embedding models is straightforward with the Sentence-Transformers library:

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('ibm-granite/granite-embedding-english-r2')

# Encode some text
documents = [
    "Granite models are designed for enterprise applications",
    "Information retrieval systems need fast and accurate embeddings",
    "Machine learning models can process natural language"
]

# Generate embeddings
embeddings = model.encode(documents)
print(f"Embedding shape: {embeddings.shape}")  # (3, 768)
```

### Semantic Search

```python
import numpy as np
from sentence_transformers import util

# Encode query and documents
query = "What are enterprise AI models?"
query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# Compute cosine similarity
similarities = util.cos_sim(query_embedding, doc_embeddings)
print(f"Similarities: {similarities}")

# Get most relevant document
best_idx = np.argmax(similarities)
print(f"Most relevant: {documents[best_idx]}")
```

### Building a Complete Retrieval System with Reranker

```python
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Load retriever and reranker
retriever = SentenceTransformer('ibm-granite/granite-embedding-english-r2')
reranker = CrossEncoder('ibm-granite/granite-embedding-reranker-english-r2')

# Your document corpus
corpus = [
    "Python is a high-level programming language",
    "Machine learning models require training data",
    "Natural language processing enables text understanding",
    "Deep learning uses neural networks with multiple layers",
    "Data science combines statistics and programming",
]

# Step 1: Encode corpus once (can be cached)
corpus_embeddings = retriever.encode(corpus, convert_to_tensor=True)

# Step 2: Retrieve top-k candidates
def search(query, top_k=20):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    
    # Find top-k with retriever
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    
    # Step 3: Rerank with cross-encoder
    cross_inp = [[query, corpus[hit['corpus_id']]] for hit in hits]
    cross_scores = reranker.predict(cross_inp)
    
    # Sort by reranker scores
    for idx, score in enumerate(cross_scores):
        hits[idx]['rerank_score'] = score
    
    hits = sorted(hits, key=lambda x: x['rerank_score'], reverse=True)
    
    return hits[:5]  # Return top 5 after reranking

# Use it
results = search("What is machine learning?")
for hit in results:
    print(f"Score: {hit['rerank_score']:.4f} | {corpus[hit['corpus_id']]}")
```

### Long Context Documents

Granite R2 handles up to 8,192 tokens, perfect for processing full documents:

```python
# Load model with long context support
model = SentenceTransformer('ibm-granite/granite-embedding-english-r2')

# Process a long document (e.g., research paper, technical documentation)
long_document = """
[Your 5000+ word document here]
This could be an entire research paper, technical manual, 
or any long-form content...
"""

# Encode the full document (no chunking needed for <8192 tokens)
doc_embedding = model.encode(long_document, show_progress_bar=True)

# Compare with shorter query
query = "What are the main findings of this research?"
query_embedding = model.encode(query)

similarity = util.cos_sim(query_embedding, doc_embedding)
print(f"Relevance score: {similarity.item():.4f}")
```

### Code Search

Granite R2 excels at code retrieval:

```python
# Code snippets corpus
code_snippets = [
    """
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    """,
    """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """
]

# Encode code
code_embeddings = model.encode(code_snippets)

# Natural language query
query = "How do I implement a binary search algorithm?"
query_embedding = model.encode(query)

# Find most relevant code
similarities = util.cos_sim(query_embedding, code_embeddings)[0]
best_match = np.argmax(similarities)

print(f"Most relevant code snippet:\n{code_snippets[best_match]}")
```

### Table Retrieval

Handle structured data with ease:

```python
# Tables in markdown format
tables = [
    """
    | Product | Q1 Revenue | Q2 Revenue |
    |---------|-----------|-----------|
    | Product A | $500K | $650K |
    | Product B | $300K | $420K |
    """,
    """
    | Employee | Department | Salary |
    |----------|-----------|--------|
    | John Doe | Engineering | $120K |
    | Jane Smith | Marketing | $95K |
    """
]

# Encode tables
table_embeddings = model.encode(tables)

# Query for specific information
query = "What was the revenue growth for our products?"
query_embedding = model.encode(query)

similarities = util.cos_sim(query_embedding, table_embeddings)[0]
best_table = np.argmax(similarities)

print(f"Most relevant table:\n{tables[best_table]}")
```

### Batch Processing for Production

For production deployments processing large volumes:

```python
from sentence_transformers import SentenceTransformer
import torch

# Load model with GPU support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('ibm-granite/granite-embedding-english-r2', device=device)

# Large batch of documents
documents = [...]  # Your thousands of documents

# Efficient batch encoding
batch_size = 128
all_embeddings = model.encode(
    documents,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_tensor=True,
    normalize_embeddings=True  # For cosine similarity
)

# Save embeddings for later use
torch.save(all_embeddings, 'document_embeddings.pt')

# Load and search later
embeddings = torch.load('document_embeddings.pt')
query_emb = model.encode(query, convert_to_tensor=True)
hits = util.semantic_search(query_emb, embeddings, top_k=10)
```

## Get Started Now

All Granite Embedding R2 models are available now on Hugging Face under Apache 2.0 license:

- [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2)
- [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- [granite-embedding-reranker-english-r2](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2)

Installation is simple:
```bash
pip install sentence-transformers
```

For technical details, architecture ablations, and comprehensive benchmark results, see our [research paper](https://arxiv.org/abs/2508.21085).

---

*The Granite Embedding R2 models represent collaborative work across IBM Research teams in multiple geographies. For questions or feedback, visit our [GitHub repository](https://github.com/ibm-granite/granite-embedding-models).*