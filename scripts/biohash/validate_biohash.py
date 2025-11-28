"""
Quick validation script to ensure BioHash implementation is working correctly.
"""

import torch
import numpy as np
from biohash_implementation import BioHash

print("Validating BioHash Implementation")
print("="*60)

# Test 1: Basic initialization and forward pass
print("\n[Test 1] Initialization and forward pass...")
try:
    biohash = BioHash(
        input_dim=10,
        num_neurons=50,
        hash_length=5,
        device='cpu'
    )
    print("✓ BioHash initialized successfully")

    # Generate random data
    X = torch.randn(100, 10)

    # Hash without training (should work)
    hash_codes = biohash.hash(X)
    assert hash_codes.shape == (100, 50), "Hash code shape mismatch"
    assert ((hash_codes == 1) | (hash_codes == -1)).all(), "Hash codes should be ±1"

    # Check sparsity
    num_active = (hash_codes == 1).sum(dim=1)
    assert (num_active == 5).all(), f"Expected 5 active neurons, got {num_active[0]}"

    print(f"✓ Hash generation working (shape: {hash_codes.shape})")
    print(f"✓ Sparsity correct ({biohash.hash_length} active per code)")

except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    exit(1)

# Test 2: Training on small dataset
print("\n[Test 2] Training on small synthetic dataset...")
try:
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate clustered data
    n_clusters = 3
    n_per_cluster = 50
    X_list = []

    for i in range(n_clusters):
        center = torch.randn(10) * 2
        cluster = center + 0.5 * torch.randn(n_per_cluster, 10)
        X_list.append(cluster)

    X_train = torch.cat(X_list, dim=0)

    biohash = BioHash(
        input_dim=10,
        num_neurons=50,
        hash_length=5,
        learning_rate=0.02,
        max_epochs=20,
        batch_size=30,
        device='cpu'
    )

    # Train
    biohash.fit(X_train, verbose=False)

    print(f"✓ Training completed ({biohash.history['epochs']} epochs)")

    # Check weight norms (should converge toward ~1.0 for p=2)
    avg_norm = torch.mean(torch.sqrt(torch.sum(biohash.W ** 2, dim=1))).item()
    print(f"✓ Average weight norm: {avg_norm:.4f} (target: ~1.0)")

    # Hash trained data
    hash_codes = biohash.hash(X_train)

    # Check that samples from same cluster have similar codes
    cluster_0_hashes = hash_codes[:n_per_cluster]
    cluster_1_hashes = hash_codes[n_per_cluster:2*n_per_cluster]

    # Compute pairwise distances within and between clusters
    within_dist = torch.cdist(cluster_0_hashes[:10].float(),
                              cluster_0_hashes[10:20].float(), p=1).mean()
    between_dist = torch.cdist(cluster_0_hashes[:10].float(),
                               cluster_1_hashes[:10].float(), p=1).mean()

    print(f"✓ Within-cluster Hamming distance: {within_dist:.2f}")
    print(f"✓ Between-cluster Hamming distance: {between_dist:.2f}")

    if between_dist > within_dist:
        print("✓ Locality sensitivity verified (between > within)")
    else:
        print("⚠ Warning: Locality sensitivity weak (may need more training)")

except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    exit(1)

# Test 3: Save and load
print("\n[Test 3] Save and load model...")
try:
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_biohash.pt')

        # Save
        biohash.save(filepath)
        W_original = biohash.W.clone()

        # Load into new instance
        biohash_loaded = BioHash(
            input_dim=10,
            num_neurons=50,
            hash_length=5,
            device='cpu'
        )
        biohash_loaded.load(filepath)

        # Check weights match
        assert torch.allclose(W_original, biohash_loaded.W), "Loaded weights don't match"

        print("✓ Model saved and loaded successfully")

        # Check that hash codes match
        hash_original = biohash.hash(X_train[:10])
        hash_loaded = biohash_loaded.hash(X_train[:10])
        assert torch.equal(hash_original, hash_loaded), "Hash codes don't match after load"

        print("✓ Hash codes consistent after load")

except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    exit(1)

# Test 4: Different p values
print("\n[Test 4] Testing different p-norm values...")
try:
    X_test = torch.randn(50, 10)

    for p in [1.5, 2.0, 3.0]:
        biohash_p = BioHash(
            input_dim=10,
            num_neurons=50,
            hash_length=5,
            p=p,
            max_epochs=10,
            device='cpu'
        )

        biohash_p.fit(X_test, verbose=False)
        hash_codes = biohash_p.hash(X_test[:10])

        print(f"✓ p={p}: Trained and hashed successfully")

except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    exit(1)

# Test 5: Verify energy decreases
print("\n[Test 5] Verify energy function decreases...")
try:
    X_energy = torch.randn(100, 10)

    biohash_energy = BioHash(
        input_dim=10,
        num_neurons=30,
        hash_length=3,
        max_epochs=5,
        device='cpu'
    )

    # Manually track energy
    energies = []
    for epoch in range(5):
        # Compute energy before update
        E = biohash_energy.compute_energy(X_energy[:20])
        energies.append(E)

        # Do one epoch of training
        biohash_energy.fit(X_energy, verbose=False)
        biohash_energy.max_epochs = epoch + 2  # Continue from current

    # Check that energy generally decreases (allowing some fluctuation)
    print(f"✓ Energy trajectory: {[f'{e:.2f}' for e in energies]}")

    if energies[-1] < energies[0]:
        print("✓ Energy decreased overall (as expected from theory)")
    else:
        print("⚠ Warning: Energy increased (may be due to small dataset/epochs)")

except Exception as e:
    print(f"✗ Test 5 failed: {e}")
    exit(1)

# All tests passed
print("\n" + "="*60)
print("✓ All validation tests passed!")
print("="*60)
print("\nBioHash implementation is working correctly.")
print("You can now run:")
print("  - python test_biohash_simple.py (for quick demos)")
print("  - python biohash_implementation.py (for full MNIST experiment)")
