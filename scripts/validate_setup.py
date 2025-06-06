#!/usr/bin/env python3
"""
Simple validation script to test that the training setup is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    try:
        from data_utils import load_tox21, split_dataset
        print("data_utils import successful")
        
        from model import GIN
        print("model import successful")
        
        from train import train_epoch, evaluate, save_checkpoint, save_model_only
        print("train utilities import successful")
        
        import torch
        import torch_geometric
        print("PyTorch and PyTorch Geometric available")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic dataset loading functionality."""
    try:
        from data_utils import load_tox21
        
        # Test loading dataset
        dataset = load_tox21(
            root="data/Tox21",
            filename="tox21.csv",
            smiles_col="smiles",
            mol_id_col="mol_id",
            cache_file="data.pt",
            recreate=False,
            auto_download=False
        )
        
        print(f"Dataset loaded successfully: {len(dataset)} molecules")
        print(f"Node features: {dataset.num_node_features}")
        
        return True
    except Exception as e:
        print(f"Dataset loading error: {e}")
        return False

def main():
    """Run validation tests."""
    print("=== Training Setup Validation ===")
    
    imports_ok = test_imports()
    if not imports_ok:
        print("\nImport tests failed. Please check your environment setup.")
        return False
    
    print("\n=== Testing Basic Functionality ===")
    functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\nAll validation tests passed! Training setup is ready.")
        print("\nYou can now run the training script:")
        print("  python scripts/train.py --epochs 50 --lr 1e-3 --hidden_channels 64")
        return True
    else:
        print("\nSome validation tests failed. Please check the setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
