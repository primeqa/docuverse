"""
Simple test script for BioHash on synthetic data and small MNIST subset.
"""

import numpy as np
import torch
from biohash_implementation import BioHash, compute_mAP
import matplotlib.pyplot as plt


def test_synthetic_circle():
    """Test BioHash on synthetic circular data (like Figure 2 in paper)."""
    print("="*60)
    print("Test 1: Synthetic Circular Data")
    print("="*60)

    # Generate data on a circle
    n_samples = 1000
    angles = np.random.uniform(-np.pi, np.pi, n_samples)
    X = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    X = torch.FloatTensor(X)

    # Create labels based on angle ranges (for evaluation)
    labels = torch.LongTensor((angles + np.pi) // (np.pi / 5))  # 10 sectors

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Data shape: {X.shape}")

    # Train BioHash
    biohash = BioHash(
        input_dim=2,
        num_neurons=20,  # m = 20
        hash_length=2,    # k = 2 (10% activity)
        p=2.0,
        delta=0.0,
        learning_rate=0.02,
        max_epochs=50,
        batch_size=50,
        device=device
    )

    biohash.fit(X.to(device), verbose=True)

    # Visualize learned weight vectors
    W = biohash.W.cpu().numpy()
    plt.figure(figsize=(8, 8))

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, s=10, c='lightblue', label='Data')

    # Plot weight vectors
    for i in range(biohash.num_neurons):
        w = W[i]
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        plt.arrow(0, 0, w[0], w[1], head_width=0.05, head_length=0.05,
                 fc='red', ec='red', linewidth=2, alpha=0.7)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'BioHash Weight Vectors (m={biohash.num_neurons}, k={biohash.hash_length})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('biohash_circle_weights.png', dpi=150)
    print("Saved visualization to biohash_circle_weights.png")

    # Test hash generation
    hash_codes = biohash.hash(X.to(device))
    print(f"\nHash codes shape: {hash_codes.shape}")
    print(f"Example hash code: {hash_codes[0].cpu().numpy()}")
    print(f"Number of active neurons: {(hash_codes[0] == 1).sum().item()}")

    # Compute mAP
    split = int(0.8 * n_samples)
    map_score = compute_mAP(
        hash_codes[split:], hash_codes[:split],
        labels[split:].to(device), labels[:split].to(device),
        R=100
    )
    print(f"\nmAP@100: {map_score:.2f}%")


def test_mnist_small():
    """Test BioHash on a small subset of MNIST."""
    print("\n" + "="*60)
    print("Test 2: MNIST Small Subset")
    print("="*60)

    from torchvision import datasets, transforms

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Use small subset (1000 samples)
    n_samples = 1000
    indices = np.random.choice(len(train_dataset), n_samples, replace=False)

    X = torch.stack([train_dataset[i][0] for i in indices])
    y = torch.tensor([train_dataset[i][1] for i in indices])

    # Split into train and query
    split = 800
    X_train = X[:split]
    y_train = y[:split]
    X_query = X[split:]
    y_query = y[split:]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Query samples: {X_query.shape[0]}")

    # Train BioHash
    k = 8
    activity = 0.05  # 5% as in paper
    m = int(k / activity)

    print(f"\nTraining with k={k}, m={m} (activity={activity*100:.1f}%)")

    biohash = BioHash(
        input_dim=784,
        num_neurons=m,
        hash_length=k,
        p=2.0,
        delta=0.0,
        learning_rate=0.02,
        max_epochs=50,
        batch_size=50,
        device=device
    )

    biohash.fit(X_train.to(device), verbose=True)

    # Generate hash codes
    hash_train = biohash.hash(X_train.to(device))
    hash_query = biohash.hash(X_query.to(device))

    # Compute mAP
    map_score = compute_mAP(hash_query, hash_train,
                           y_query.to(device), y_train.to(device))

    print(f"\nResults:")
    print(f"  mAP@All: {map_score:.2f}%")

    # Test retrieval example
    print("\nRetrieval Example:")
    query_idx = 0
    query_label = y_query[query_idx].item()
    print(f"Query digit: {query_label}")

    query_hash = hash_query[query_idx:query_idx+1]
    distances = torch.cdist(query_hash.float(), hash_train.float(), p=1) / 2
    distances = distances.squeeze()
    sorted_indices = torch.argsort(distances)

    print("Top 5 retrievals:")
    for rank, idx in enumerate(sorted_indices[:5]):
        retrieved_label = y_train[idx].item()
        dist = distances[idx].item()
        match = "✓" if retrieved_label == query_label else "✗"
        print(f"  Rank {rank+1}: Digit {retrieved_label}, "
              f"Hamming dist={dist:.0f} {match}")


def test_hash_properties():
    """Test hash code properties."""
    print("\n" + "="*60)
    print("Test 3: Hash Code Properties")
    print("="*60)

    # Generate random data
    n_samples = 500
    input_dim = 50
    X = torch.randn(n_samples, input_dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train BioHash
    biohash = BioHash(
        input_dim=input_dim,
        num_neurons=100,
        hash_length=10,
        device=device
    )

    biohash.fit(X.to(device), verbose=False)

    # Test properties
    hash_codes = biohash.hash(X.to(device))

    print(f"\nHash code properties:")
    print(f"  Shape: {hash_codes.shape}")
    print(f"  Expected active neurons per code: {biohash.hash_length}")
    print(f"  Actual active neurons per code: {(hash_codes == 1).sum(dim=1).float().mean().item():.1f}")
    print(f"  Activity level: {(hash_codes == 1).float().mean().item()*100:.2f}%")

    # Test locality sensitivity
    # Nearby points should have similar hash codes
    x1 = X[0].to(device)
    x2 = X[0].to(device) + 0.1 * torch.randn_like(X[0].to(device))  # Nearby point
    x3 = X[1].to(device)  # Different point

    h1 = biohash.hash(x1.unsqueeze(0))
    h2 = biohash.hash(x2.unsqueeze(0))
    h3 = biohash.hash(x3.unsqueeze(0))

    dist_nearby = torch.cdist(h1.float(), h2.float(), p=1).item() / 2
    dist_far = torch.cdist(h1.float(), h3.float(), p=1).item() / 2

    print(f"\nLocality sensitivity test:")
    print(f"  Input similarity (cosine): x1-x2={torch.cosine_similarity(x1, x2, dim=0).item():.3f}, "
          f"x1-x3={torch.cosine_similarity(x1, x3, dim=0).item():.3f}")
    print(f"  Hamming distance: nearby={dist_nearby:.0f}, far={dist_far:.0f}")
    print(f"  Locality preserved: {dist_nearby < dist_far}")


if __name__ == '__main__':
    print("BioHash Simple Tests")
    print("Implementation of 'Bio-Inspired Hashing' (Ryali et al., ICML 2020)\n")

    # Run tests
    test_synthetic_circle()
    test_mnist_small()
    test_hash_properties()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
