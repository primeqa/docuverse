#!/usr/bin/env python3
"""
Example demonstrating array field path usage with realistic RAG data.
"""

import json
import tempfile
import os

# Create example RAG data with multiple retrieved documents
sample_data = [
    {
        "question": "What is machine learning?",
        "retrieved_docs": [
            {"title": "ML Intro", "text": "Machine learning is a subset of AI that enables systems to learn from data."},
            {"title": "ML Applications", "text": "ML is used in recommendation systems, image recognition, and NLP."},
            {"title": "Deep Learning", "text": "Deep learning is a type of ML using neural networks with multiple layers."}
        ],
        "answer": "Machine learning is a subset of artificial intelligence."
    },
    {
        "question": "What are neural networks?",
        "retrieved_docs": [
            {"title": "Neural Nets", "text": "Neural networks are computing systems inspired by biological neural networks."},
            {"title": "Architecture", "text": "They consist of interconnected nodes organized in layers."}
        ],
        "answer": "Neural networks are computing systems inspired by the brain."
    },
    {
        "question": "What is NLP?",
        "retrieved_docs": [
            {"title": "NLP Overview", "text": "Natural Language Processing enables computers to understand human language."},
            {"title": "NLP Tasks", "text": "Common NLP tasks include translation, sentiment analysis, and question answering."},
            {"title": "Transformers", "text": "Modern NLP relies heavily on transformer architectures like BERT and GPT."}
        ],
        "answer": "NLP is a field focused on human-computer language interaction."
    }
]

# Write to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in sample_data:
        f.write(json.dumps(item) + '\n')
    temp_file = f.name

print("="*80)
print("ARRAY FIELD PATH DEMONSTRATION")
print("="*80)
print(f"\nCreated example file: {temp_file}")
print(f"Contains {len(sample_data)} questions with multiple retrieved documents each\n")

# Show example data structure
print("Example data structure:")
print("-"*80)
print(json.dumps(sample_data[0], indent=2))
print("-"*80)

print("\n" + "="*80)
print("FIELD PATH EXAMPLES")
print("="*80)

# Import the reading function
from benchmark_embedding_timing import read_jsonl_file

# Example 1: Extract all questions
print("\n1. Extract questions:")
print("   Field path: 'question'")
texts = read_jsonl_file(temp_file, field_path="question")
print(f"   → Extracted {len(texts)} texts")
for i, text in enumerate(texts, 1):
    print(f"   {i}. {text[:60]}...")

# Example 2: Extract all retrieved document texts (wildcard)
print("\n2. Extract ALL retrieved documents using wildcard:")
print("   Field path: 'retrieved_docs[*].text'")
texts = read_jsonl_file(temp_file, field_path="retrieved_docs[*].text")
print(f"   → Extracted {len(texts)} texts (3 + 2 + 3 = 8 total)")
for i, text in enumerate(texts, 1):
    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"   {i}. {preview}")

# Example 3: Extract only first retrieved document
print("\n3. Extract FIRST retrieved document only:")
print("   Field path: 'retrieved_docs[0].text'")
texts = read_jsonl_file(temp_file, field_path="retrieved_docs[0].text")
print(f"   → Extracted {len(texts)} texts (one per question)")
for i, text in enumerate(texts, 1):
    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"   {i}. {preview}")

# Example 4: Extract document titles
print("\n4. Extract all document titles:")
print("   Field path: 'retrieved_docs[*].title'")
texts = read_jsonl_file(temp_file, field_path="retrieved_docs[*].title")
print(f"   → Extracted {len(texts)} titles")
for i, text in enumerate(texts, 1):
    print(f"   {i}. {text}")

# Example 5: Extract second document from each question
print("\n5. Extract SECOND retrieved document (where available):")
print("   Field path: 'retrieved_docs[1].text'")
texts = read_jsonl_file(temp_file, field_path="retrieved_docs[1].text")
print(f"   → Extracted {len(texts)} texts")
for i, text in enumerate(texts, 1):
    preview = text[:60] + "..." if len(text) > 60 else text
    print(f"   {i}. {preview}")

print("\n" + "="*80)
print("PRACTICAL USAGE")
print("="*80)

print("""
Now you can benchmark these different extraction patterns:

# Benchmark all retrieved documents (realistic load):
python benchmark_embedding_timing.py \\
    --input_file """ + temp_file + """ \\
    --field_path "retrieved_docs[*].text" \\
    --batch_sizes 1,8,16

# Benchmark only top retrieved document:
python benchmark_embedding_timing.py \\
    --input_file """ + temp_file + """ \\
    --field_path "retrieved_docs[0].text" \\
    --batch_sizes 1,8,16

# Benchmark questions:
python benchmark_embedding_timing.py \\
    --input_file """ + temp_file + """ \\
    --field_path "question" \\
    --batch_sizes 1,8,16
""")

print("="*80)
print(f"\nExample file will be cleaned up automatically.")
print(f"To keep it for testing, the file is at: {temp_file}")

# Optional: Clean up
response = input("\nDelete example file? [y/N]: ")
if response.lower() == 'y':
    os.unlink(temp_file)
    print("✓ Example file deleted")
else:
    print(f"✓ Example file kept at: {temp_file}")
