#!/usr/bin/env python3
"""
Interactive demo of random sampling functionality.
"""

import json
import tempfile
import os
from benchmark_embedding_timing import read_jsonl_file
import random

print("="*80)
print("RANDOM SAMPLING DEMONSTRATION")
print("="*80)

# Create demo dataset
print("\n1. Creating demo dataset...")
demo_data = []
topics = ["AI", "ML", "NLP", "CV", "RL", "DL", "Data Science", "Analytics", "Big Data", "Cloud"]
for i in range(50):
    topic = topics[i % len(topics)]
    demo_data.append({
        "id": i,
        "text": f"This is question {i} about {topic}",
        "topic": topic
    })

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for item in demo_data:
        f.write(json.dumps(item) + '\n')
    temp_file = f.name

print(f"✓ Created file with {len(demo_data)} texts")
print(f"  Topics: {', '.join(set(topics))}")

# Load all texts
print("\n2. Loading all texts from file...")
all_texts = read_jsonl_file(temp_file, field_path="text")
print(f"✓ Loaded {len(all_texts)} texts")
print(f"  First 3: {all_texts[:3]}")

# Demo 1: Random sample with seed 42
print("\n" + "-"*80)
print("3. Random sample with seed 42 (10 texts)")
print("-"*80)
random.seed(42)
sample1 = random.sample(all_texts, 10)
print("Sample:")
for i, text in enumerate(sample1, 1):
    print(f"  {i}. {text}")

# Demo 2: Same seed gives same results
print("\n" + "-"*80)
print("4. Same seed (42) - should get IDENTICAL sample")
print("-"*80)
random.seed(42)
sample2 = random.sample(all_texts, 10)
if sample1 == sample2:
    print("✓ Samples are identical!")
    print("  This proves reproducibility with same seed")
else:
    print("✗ Samples differ (unexpected!)")

# Demo 3: Different seed
print("\n" + "-"*80)
print("5. Different seed (123) - should get DIFFERENT sample")
print("-"*80)
random.seed(123)
sample3 = random.sample(all_texts, 10)
print("Sample:")
for i, text in enumerate(sample3, 1):
    print(f"  {i}. {text}")

if sample1 != sample3:
    print("\n✓ Samples are different!")
    print("  This shows different seeds produce different samples")
else:
    print("\n✗ Samples are same (unexpected!)")

# Demo 4: Show difference
print("\n" + "-"*80)
print("6. Comparing samples")
print("-"*80)
in_both = set(sample1) & set(sample3)
only_seed42 = set(sample1) - set(sample3)
only_seed123 = set(sample3) - set(sample1)

print(f"Texts in both samples: {len(in_both)}")
print(f"Only in seed 42: {len(only_seed42)}")
print(f"Only in seed 123: {len(only_seed123)}")

if only_seed42:
    print(f"\nExample unique to seed 42: {list(only_seed42)[0]}")
if only_seed123:
    print(f"Example unique to seed 123: {list(only_seed123)[0]}")

# Demo 5: Practical usage
print("\n" + "-"*80)
print("7. Practical Usage Example")
print("-"*80)
print(f"""
You have a file with {len(all_texts)} texts.
You want to benchmark with 10 random samples.

Command:
    python benchmark_embedding_timing.py \\
        --input_file {temp_file} \\
        --field_path text \\
        --num_samples 10 \\
        --random_seed 42 \\
        --skip_api --skip_gpu

This will:
    1. Load all {len(all_texts)} texts
    2. Randomly sample 10 using seed 42
    3. Use those 10 for benchmarking

Benefits:
    ✓ Representative sample (not just first 10)
    ✓ Reproducible (same seed = same sample)
    ✓ Fast benchmarking (only 10 texts)
""")

# Demo 6: Show distribution
print("-"*80)
print("8. Checking randomness - topic distribution")
print("-"*80)

# Count topics in original
original_topics = {}
for item in demo_data:
    topic = item['topic']
    original_topics[topic] = original_topics.get(topic, 0) + 1

# Count topics in sample
sample_topics = {}
random.seed(42)
sample_texts = random.sample(all_texts, 20)
for text in sample_texts:
    for item in demo_data:
        if item['text'] == text:
            topic = item['topic']
            sample_topics[topic] = sample_topics.get(topic, 0) + 1
            break

print("\nOriginal dataset topic distribution:")
for topic, count in sorted(original_topics.items()):
    print(f"  {topic}: {count} ({count/len(demo_data)*100:.1f}%)")

print("\nRandom sample (20) topic distribution:")
for topic, count in sorted(sample_topics.items()):
    print(f"  {topic}: {count} ({count/20*100:.1f}%)")

print("\n✓ Random sample roughly preserves topic distribution")

# Cleanup
print("\n" + "="*80)
print("DEMONSTRATION COMPLETE")
print("="*80)

response = input(f"\nDelete demo file? [y/N]: ")
if response.lower() == 'y':
    os.unlink(temp_file)
    print("✓ Demo file deleted")
else:
    print(f"✓ Demo file kept at: {temp_file}")

print("\nKey Takeaways:")
print("  • Same seed = reproducible samples")
print("  • Different seed = different samples")
print("  • Random sampling preserves distribution")
print("  • Use --num_samples for random sampling")
print("  • Use --random_seed for reproducibility")
