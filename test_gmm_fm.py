"""
Quick test script for GMM-FM implementation
Tests basic functionality without full training setup
"""

import numpy as np
import os
import sys

print("="*70)
print("GMM-FM Implementation Test")
print("="*70)

# Test 1: GMM Prior utilities
print("\n[Test 1] Testing GMM Prior utilities...")
try:
    from utils.gmm_prior import load_gmm_prior, save_gmm_prior

    # Load dummy GMM
    prior = load_gmm_prior('gmm_cache/prior.npz', verbose=True)
    print("✓ Successfully loaded GMM prior")

    # Check parameters
    assert prior.pi.shape[0] == 10, "Wrong K"
    assert prior.mu.shape[1] == 4096, "Wrong D"
    print(f"✓ GMM parameters validated: K={prior.pi.shape[0]}, D={prior.mu.shape[1]}")

except Exception as e:
    print(f"✗ Test 1 failed: {e}")
    sys.exit(1)

# Test 2: GMM EM utilities (just imports)
print("\n[Test 2] Testing GMM EM module...")
try:
    from utils.gmm_em import GMMParams, log_joint_diag, e_step, m_step

    print("✓ Successfully imported GMM EM functions")

except Exception as e:
    print(f"✗ Test 2 failed: {e}")
    sys.exit(1)

# Test 3: Targets GMM-FM (just imports and basic checks)
print("\n[Test 3] Testing GMM-FM targets module...")
try:
    from targets_gmm_fm import get_targets, reset_gmm_cache

    print("✓ Successfully imported GMM-FM targets")

except Exception as e:
    print(f"✗ Test 3 failed: {e}")
    sys.exit(1)

# Test 4: Check if train.py has gmm-fm routing
print("\n[Test 4] Checking train.py modifications...")
try:
    with open('train.py', 'r') as f:
        train_content = f.read()

    # Check for GMM flags
    assert 'gmm_K' in train_content, "Missing gmm_K flag"
    assert 'gmm_assign_mode' in train_content, "Missing gmm_assign_mode flag"
    print("✓ GMM config flags found in train.py")

    # Check for GMM preprocessing
    assert 'GMM-Prior Preprocessing' in train_content, "Missing GMM preprocessing"
    print("✓ GMM preprocessing code found in train.py")

    # Check for routing
    assert "FLAGS.model['train_type'] == 'gmm-fm'" in train_content, "Missing gmm-fm routing"
    print("✓ GMM-FM routing found in train.py")

except Exception as e:
    print(f"✗ Test 4 failed: {e}")
    sys.exit(1)

# Test 5: Check helper files modifications
print("\n[Test 5] Checking helper files modifications...")
try:
    # Check helper_inference.py
    with open('helper_inference.py', 'r') as f:
        inf_content = f.read()
    assert "FLAGS.model['train_type'] == 'gmm-fm'" in inf_content, "Missing gmm-fm in helper_inference.py"
    assert 'sample_gmm_unconditional' in inf_content, "Missing GMM sampling in helper_inference.py"
    print("✓ helper_inference.py modified correctly")

    # Check helper_eval.py
    with open('helper_eval.py', 'r') as f:
        eval_content = f.read()
    assert "FLAGS.model['train_type'] == 'gmm-fm'" in eval_content, "Missing gmm-fm in helper_eval.py"
    assert 'sample_gmm_unconditional' in eval_content, "Missing GMM sampling in helper_eval.py"
    print("✓ helper_eval.py modified correctly")

except Exception as e:
    print(f"✗ Test 5 failed: {e}")
    sys.exit(1)

# Test 6: Verify file structure
print("\n[Test 6] Verifying file structure...")
required_files = [
    'utils/gmm_em.py',
    'utils/gmm_prior.py',
    'targets_gmm_fm.py',
    'gmm_cache/prior.npz',
]

all_exist = True
for filepath in required_files:
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
    else:
        print(f"✗ {filepath} NOT FOUND")
        all_exist = False

if not all_exist:
    print("✗ Test 6 failed: Missing files")
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nImplementation is ready. Next steps:")
print("1. Test with real training: python train.py --model.train_type=gmm-fm --max_steps=100")
print("2. Monitor W&B dashboard for GMM preprocessing logs")
print("3. Experiment with different assignment modes (soft_sample, moment, hard)")
print("4. Adjust GMM hyperparameters (K, subset, temperature)")
print("="*70)
