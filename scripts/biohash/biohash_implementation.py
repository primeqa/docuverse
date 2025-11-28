"""
BioHash: Bio-Inspired Hashing for Unsupervised Similarity Search

Implementation of the algorithm from:
"Bio-Inspired Hashing for Unsupervised Similarity Search"
Ryali et al., ICML 2020

This implementation includes:
- Biologically plausible learning dynamics (Equation 1)
- k-WTA hash code generation (Equation 6)
- Training and testing on MNIST/CIFAR-10
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import pickle
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class BioHash:
    """
    BioHash implementation using biologically plausible learning dynamics.

    The learning dynamics for each neuron μ is:
    τ dW_μi/dt = g[Rank(⟨W_μ, x⟩_μ)] (x_i - ⟨W_μ, x⟩_μ W_μi)

    where:
    - g[1] = 1 (Hebbian update for winner)
    - g[r] = -Δ (Anti-Hebbian update for rank r)
    - g[other] = 0 (no update for others)
    """

    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        hash_length: int,
        p: float = 2.0,
        delta: float = 0.0,
        r: int = 2,
        learning_rate: float = 0.02,
        max_epochs: int = 100,
        batch_size: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize BioHash.

        Args:
            input_dim: Dimensionality of input data (d)
            num_neurons: Number of hash neurons (m), typically m >> d
            hash_length: Number of active neurons (k)
            p: Exponent for p-norm (p=2 for spherical K-means)
            delta: Anti-Hebbian weight (Δ)
            r: Rank for anti-Hebbian update
            learning_rate: Initial learning rate (ε₀)
            max_epochs: Maximum training epochs
            batch_size: Mini-batch size
            device: 'cuda' or 'cpu'
        """
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.hash_length = hash_length
        self.p = p
        self.delta = delta
        self.r = r
        self.learning_rate_init = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device

        # Activity level (sparsity)
        self.activity = hash_length / num_neurons

        # Initialize weights W ∈ R^(m×d) from standard normal
        self.W = torch.randn(num_neurons, input_dim, device=device)

        # Data statistics for centering
        self.data_mean = None

        # Training history
        self.history = {
            'energy': [],
            'weight_norms': [],
            'epochs': 0
        }

    def compute_inner_product(self, W_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute generalized inner product ⟨W_μ, x⟩_μ with η^μ_ij = |W_μi|^(p-2) δ_ij.

        For p=2, this simplifies to the standard inner product.
        """
        if self.p == 2.0:
            return torch.sum(W_mu * x, dim=-1)
        else:
            # η^μ_ij = |W_μi|^(p-2) δ_ij (diagonal matrix)
            eta = torch.abs(W_mu) ** (self.p - 2)
            return torch.sum(eta * W_mu * x, dim=-1)

    def compute_norm(self, W_mu: torch.Tensor) -> torch.Tensor:
        """Compute ⟨W_μ, W_μ⟩_μ."""
        if self.p == 2.0:
            return torch.sum(W_mu * W_mu, dim=-1)
        else:
            eta = torch.abs(W_mu) ** (self.p - 2)
            return torch.sum(eta * W_mu * W_mu, dim=-1)

    def update_weights(self, X: torch.Tensor, learning_rate: float):
        """
        Perform one weight update step using the learning dynamics from Equation (1).

        Args:
            X: Batch of data points (batch_size × d)
            learning_rate: Current learning rate
        """
        batch_size = X.shape[0]

        for x in X:
            # Compute inner products for all neurons
            inner_products = torch.zeros(self.num_neurons, device=self.device)
            for mu in range(self.num_neurons):
                inner_products[mu] = self.compute_inner_product(self.W[mu], x)

            # Rank neurons by inner product (highest to lowest)
            ranks = torch.argsort(inner_products, descending=True)

            # Create g vector
            g = torch.zeros(self.num_neurons, device=self.device)
            winner_idx = ranks[0]
            g[winner_idx] = 1.0

            if self.r <= self.num_neurons:
                rank_r_idx = ranks[self.r - 1]
                g[rank_r_idx] = -self.delta

            # Update weights for each neuron
            for mu in range(self.num_neurons):
                if g[mu] != 0:
                    inner_prod = inner_products[mu]

                    # Compute update: g[μ] * (x - ⟨W_μ, x⟩_μ W_μ)
                    if self.p == 2.0:
                        update = g[mu] * (x - inner_prod * self.W[mu])
                    else:
                        # For general p, need to account for η
                        eta = torch.abs(self.W[mu]) ** (self.p - 2)
                        weighted_W = eta * self.W[mu]
                        update = g[mu] * (x - inner_prod * weighted_W)

                    self.W[mu] = self.W[mu] + learning_rate * update

    def compute_energy(self, X: torch.Tensor) -> float:
        """
        Compute energy function (Equation 3) for monitoring training progress.
        """
        energy = 0.0

        with torch.no_grad():
            for x in X:
                # Compute inner products for all neurons
                inner_products = torch.zeros(self.num_neurons, device=self.device)
                norms = torch.zeros(self.num_neurons, device=self.device)

                for mu in range(self.num_neurons):
                    inner_products[mu] = self.compute_inner_product(self.W[mu], x)
                    norms[mu] = self.compute_norm(self.W[mu])

                # Rank neurons
                ranks = torch.argsort(inner_products, descending=True)

                # Create g vector
                g = torch.zeros(self.num_neurons, device=self.device)
                winner_idx = ranks[0]
                g[winner_idx] = 1.0

                if self.r <= self.num_neurons:
                    rank_r_idx = ranks[self.r - 1]
                    g[rank_r_idx] = -self.delta

                # Compute energy contribution
                for mu in range(self.num_neurons):
                    if g[mu] != 0 and norms[mu] > 0:
                        term = inner_products[mu] / (norms[mu] ** ((self.p - 1) / self.p))
                        energy -= g[mu] * term

        return energy / len(X)

    def fit(self, X_train: torch.Tensor, verbose: bool = True):
        """
        Train BioHash on data.

        Args:
            X_train: Training data (n × d)
            verbose: Whether to print progress
        """
        # Center the data
        self.data_mean = X_train.mean(dim=0)
        X_centered = X_train - self.data_mean

        # Create data loader
        dataset = TensorDataset(X_centered)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if verbose:
            print(f"Training BioHash: {self.num_neurons} neurons, k={self.hash_length}")
            print(f"Activity level: {self.activity*100:.2f}%")

        # Training loop
        for epoch in range(self.max_epochs):
            # Decay learning rate
            lr = self.learning_rate_init * (1 - epoch / self.max_epochs)

            # Train on batches
            for (batch,) in dataloader:
                batch = batch.to(self.device)
                self.update_weights(batch, lr)

            # Monitor progress
            if epoch % 10 == 0 or epoch == self.max_epochs - 1:
                avg_norm = torch.mean(torch.sqrt(torch.sum(self.W ** 2, dim=1))).item()
                self.history['weight_norms'].append(avg_norm)
                self.history['epochs'] = epoch + 1

                if verbose:
                    print(f"Epoch {epoch+1}/{self.max_epochs}, "
                          f"LR: {lr:.5f}, Avg Weight Norm: {avg_norm:.4f}")

                # Check convergence (average norm ≈ 1.06)
                if abs(avg_norm - 1.06) < 0.01:
                    if verbose:
                        print(f"Converged at epoch {epoch+1}")
                    break

        if verbose:
            print("Training completed!")

    def hash(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate hash codes using k-WTA (Equation 6).

        Args:
            X: Data to hash (n × d)

        Returns:
            Hash codes (n × m) with values in {-1, +1}
        """
        # Center using training mean
        if self.data_mean is not None:
            X_centered = X - self.data_mean
        else:
            X_centered = X

        n = X_centered.shape[0]
        hash_codes = torch.ones(n, self.num_neurons, device=self.device) * (-1)

        with torch.no_grad():
            for i, x in enumerate(X_centered):
                # Compute inner products
                inner_products = torch.zeros(self.num_neurons, device=self.device)
                for mu in range(self.num_neurons):
                    inner_products[mu] = self.compute_inner_product(self.W[mu], x)

                # k-WTA: activate top k neurons
                top_k_indices = torch.topk(inner_products, self.hash_length)[1]
                hash_codes[i, top_k_indices] = 1

        return hash_codes

    def hamming_distance(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Compute Hamming distance between hash codes.

        For sparse codes, we use the fact that we store indices of active neurons.
        """
        # h1: (n1 × m), h2: (n2 × m)
        # Hamming distance = number of positions where they differ
        distances = torch.cdist(h1.float(), h2.float(), p=1) / 2
        return distances.int()

    def save(self, filepath: str):
        """Save model to file."""
        state = {
            'W': self.W.cpu(),
            'data_mean': self.data_mean.cpu() if self.data_mean is not None else None,
            'config': {
                'input_dim': self.input_dim,
                'num_neurons': self.num_neurons,
                'hash_length': self.hash_length,
                'p': self.p,
                'delta': self.delta,
                'r': self.r
            },
            'history': self.history
        }
        torch.save(state, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file."""
        state = torch.load(filepath, map_location=self.device)
        self.W = state['W'].to(self.device)
        self.data_mean = state['data_mean'].to(self.device) if state['data_mean'] is not None else None
        self.history = state['history']
        print(f"Model loaded from {filepath}")


def compute_mAP(hash_codes_query: torch.Tensor, hash_codes_db: torch.Tensor,
                labels_query: torch.Tensor, labels_db: torch.Tensor,
                R: Optional[int] = None) -> float:
    """
    Compute Mean Average Precision (mAP) for retrieval.

    Args:
        hash_codes_query: Hash codes for queries (n_q × m)
        hash_codes_db: Hash codes for database (n_db × m)
        labels_query: Ground truth labels for queries (n_q,)
        labels_db: Ground truth labels for database (n_db,)
        R: Number of retrievals (None for all)

    Returns:
        mAP score
    """
    n_queries = hash_codes_query.shape[0]
    average_precisions = []

    for i in range(n_queries):
        # Compute Hamming distances
        query_hash = hash_codes_query[i:i+1]
        distances = torch.cdist(query_hash.float(), hash_codes_db.float(), p=1) / 2
        distances = distances.squeeze()

        # Sort by distance
        sorted_indices = torch.argsort(distances)

        # Determine relevant items (same label)
        query_label = labels_query[i]
        relevant = (labels_db[sorted_indices] == query_label).cpu().numpy()

        # Compute Average Precision
        if R is not None:
            relevant = relevant[:R]

        num_relevant = relevant.sum()
        if num_relevant == 0:
            continue

        precisions = []
        num_retrieved_relevant = 0

        for j, rel in enumerate(relevant):
            if rel:
                num_retrieved_relevant += 1
                precision = num_retrieved_relevant / (j + 1)
                precisions.append(precision)

        if len(precisions) > 0:
            ap = np.mean(precisions)
            average_precisions.append(ap)

    return np.mean(average_precisions) * 100  # Return as percentage


def load_mnist(data_dir: str = './data') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and preprocess MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Separate training and query sets as per paper
    # Use 100 images per class (1000 total) as query set
    np.random.seed(42)
    query_indices = []
    for digit in range(10):
        digit_indices = np.where(train_dataset.targets.numpy() == digit)[0]
        selected = np.random.choice(digit_indices, size=100, replace=False)
        query_indices.extend(selected.tolist())

    query_indices = set(query_indices)
    train_indices = [i for i in range(len(train_dataset)) if i not in query_indices]

    # Create datasets
    X_train = torch.stack([train_dataset[i][0] for i in train_indices])
    y_train = torch.tensor([train_dataset[i][1] for i in train_indices])

    X_query = torch.stack([train_dataset[i][0] for i in query_indices])
    y_query = torch.tensor([train_dataset[i][1] for i in query_indices])

    return X_train, y_train, X_query, y_query


def experiment_mnist(hash_lengths=[2, 4, 8, 16, 32], activity=0.05):
    """Run BioHash experiment on MNIST."""
    print("="*60)
    print("BioHash Experiment on MNIST")
    print("="*60)

    # Load data
    print("\nLoading MNIST dataset...")
    X_train, y_train, X_query, y_query = load_mnist()

    input_dim = X_train.shape[1]  # 784
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Query set: {X_query.shape[0]} samples")

    results = {}

    for k in hash_lengths:
        print(f"\n{'='*60}")
        print(f"Hash Length k = {k}")
        print(f"{'='*60}")

        # Determine m from activity level: m * activity = k
        m = int(k / activity)

        # Initialize and train BioHash
        biohash = BioHash(
            input_dim=input_dim,
            num_neurons=m,
            hash_length=k,
            p=2.0,
            delta=0.0,
            r=2,
            learning_rate=0.02,
            max_epochs=100,
            batch_size=100,
            device=device
        )

        # Train
        start_time = time.time()
        biohash.fit(X_train.to(device), verbose=True)
        train_time = time.time() - start_time

        # Generate hash codes
        print("\nGenerating hash codes...")
        hash_train = biohash.hash(X_train.to(device))
        hash_query = biohash.hash(X_query.to(device))

        # Compute mAP@All
        print("Computing mAP...")
        map_score = compute_mAP(hash_query, hash_train,
                               y_query.to(device), y_train.to(device))

        results[k] = {
            'mAP': map_score,
            'train_time': train_time,
            'm': m
        }

        print(f"\nResults for k={k}:")
        print(f"  m (neurons): {m}")
        print(f"  mAP@All: {map_score:.2f}%")
        print(f"  Training time: {train_time:.2f}s")

        # Save model
        biohash.save(f'biohash_mnist_k{k}.pt')

    # Summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(f"{'k':<10}{'m':<10}{'mAP@All (%)':<15}{'Time (s)':<10}")
    print("-"*60)
    for k in hash_lengths:
        r = results[k]
        print(f"{k:<10}{r['m']:<10}{r['mAP']:<15.2f}{r['train_time']:<10.2f}")

    return results


def visualize_hash_codes(biohash: BioHash, X: torch.Tensor, y: torch.Tensor,
                         save_path: str = 'hash_visualization.png'):
    """Visualize hash code patterns using t-SNE."""
    from sklearn.manifold import TSNE

    # Generate hash codes
    hash_codes = biohash.hash(X.to(biohash.device)).cpu().numpy()
    labels = y.numpy()

    # Subsample for visualization
    n_samples = min(5000, len(hash_codes))
    indices = np.random.choice(len(hash_codes), n_samples, replace=False)
    hash_subset = hash_codes[indices]
    labels_subset = labels[indices]

    # t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedding = tsne.fit_transform(hash_subset)

    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                         c=labels_subset, cmap='tab10', s=10, alpha=0.6)
    plt.colorbar(scatter, label='Digit')
    plt.title(f'BioHash Codes (k={biohash.hash_length}, m={biohash.num_neurons})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == '__main__':
    # Run experiment
    results = experiment_mnist(hash_lengths=[2, 4, 8, 16, 32])

    # Visualize for k=16
    print("\nGenerating visualization for k=16...")
    X_train, y_train, X_query, y_query = load_mnist()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    biohash = BioHash(
        input_dim=784,
        num_neurons=320,  # for 5% activity with k=16
        hash_length=16,
        device=device
    )
    biohash.load('biohash_mnist_k16.pt')

    # Use subset for visualization
    X_vis = torch.cat([X_train[:5000], X_query[:1000]])
    y_vis = torch.cat([y_train[:5000], y_query[:1000]])
    visualize_hash_codes(biohash, X_vis, y_vis)

    print("\nExperiment completed!")
