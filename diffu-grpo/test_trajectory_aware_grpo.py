#!/usr/bin/env python3
"""
Test script for TrajectoryAwareGRPOLoss implementation.

This script performs basic tests to ensure the trajectory-aware GRPO implementation
is working correctly and integrates properly with the existing codebase.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all necessary imports work correctly."""
    print("Testing imports...")
    
    try:
        from core.grpo_loss import TrajectoryAwareGRPOLoss
        from core.masking_strategy import DiffusionMaskingStrategy
        from core.memory_manager import MemoryManager
        from trajectory_aware_grpo_config import TrajectoryAwareGRPOConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_trajectory_aware_grpo_loss():
    """Test TrajectoryAwareGRPOLoss instantiation and basic functionality."""
    print("\nTesting TrajectoryAwareGRPOLoss...")
    
    try:
        from core.grpo_loss import TrajectoryAwareGRPOLoss
        from core.masking_strategy import DiffusionMaskingStrategy
        from core.memory_manager import MemoryManager
        
        # Create required components
        masking_strategy = DiffusionMaskingStrategy()
        memory_manager = MemoryManager()
        
        # Create TrajectoryAwareGRPOLoss instance
        trajectory_loss = TrajectoryAwareGRPOLoss(
            masking_strategy=masking_strategy,
            memory_manager=memory_manager,
            importance_weight_normalization="softmax",
            per_step_kl_penalty=True,
            numerical_stability_eps=1e-8,
            max_importance_weight=10.0
        )
        
        print("✓ TrajectoryAwareGRPOLoss instantiated successfully")
        
        # Test parameter updates
        trajectory_loss.update_trajectory_params(
            importance_weight_normalization="clamp",
            max_importance_weight=5.0
        )
        print("✓ Parameter updates work")
        
        # Test performance stats
        stats = trajectory_loss.get_performance_stats()
        assert 'importance_weight_normalization' in stats
        assert 'trajectory_cache_size' in stats
        print("✓ Performance stats work")
        
        return True
        
    except Exception as e:
        print(f"✗ TrajectoryAwareGRPOLoss test failed: {e}")
        return False

def test_configuration():
    """Test configuration classes."""
    print("\nTesting configuration...")
    
    try:
        from trajectory_aware_grpo_config import (
            TrajectoryAwareGRPOConfig,
            get_memory_efficient_config,
            get_high_accuracy_config,
            get_gsm8k_config
        )
        
        # Test basic config
        config = TrajectoryAwareGRPOConfig()
        config.validate_config()
        print("✓ Basic configuration works")
        
        # Test factory functions
        mem_config = get_memory_efficient_config()
        print("✓ Memory efficient config works")
        
        acc_config = get_high_accuracy_config()
        print("✓ High accuracy config works")
        
        gsm_config = get_gsm8k_config()
        print("✓ GSM8K config works")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_tensor_operations():
    """Test tensor operations with dummy data."""
    print("\nTesting tensor operations...")
    
    try:
        from core.grpo_loss import TrajectoryAwareGRPOLoss
        from core.masking_strategy import DiffusionMaskingStrategy
        from core.memory_manager import MemoryManager
        
        device = torch.device("cpu")  # Use CPU for testing
        batch_size = 2
        seq_len = 10
        
        # Create dummy data
        per_step_loss = torch.rand(batch_size, seq_len, device=device)
        total_loss = per_step_loss.sum(dim=1, keepdim=True)
        completion_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create loss instance
        masking_strategy = DiffusionMaskingStrategy()
        memory_manager = MemoryManager()
        trajectory_loss = TrajectoryAwareGRPOLoss(
            masking_strategy=masking_strategy,
            memory_manager=memory_manager
        )
        
        # Test importance weight calculation
        importance_weights = trajectory_loss._calculate_importance_weights(
            per_step_loss, total_loss, completion_mask
        )
        
        assert importance_weights.shape == (batch_size, seq_len)
        assert not torch.isnan(importance_weights).any()
        assert not torch.isinf(importance_weights).any()
        print("✓ Importance weight calculation works")
        
        # Test advantage calculation
        trajectory_rewards = torch.tensor([1.5, -0.5], device=device)
        advantages = trajectory_loss._step3_advantage_calculation(trajectory_rewards)
        
        assert advantages.shape == (batch_size,)
        assert not torch.isnan(advantages).any()
        assert not torch.isinf(advantages).any()
        print("✓ Advantage calculation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Tensor operations test failed: {e}")
        return False

def test_trainer_integration():
    """Test that the trainer can be imported with trajectory-aware loss."""
    print("\nTesting trainer integration...")
    
    try:
        from improved_diffu_grpo_trainer import ImprovedDiffuGRPOTrainer
        print("✓ Trainer imports successfully with trajectory-aware loss")
        
        # Check that TrajectoryAwareGRPOLoss is in the imports
        import improved_diffu_grpo_trainer
        assert hasattr(improved_diffu_grpo_trainer, 'TrajectoryAwareGRPOLoss')
        print("✓ TrajectoryAwareGRPOLoss is available in trainer")
        
        return True
        
    except Exception as e:
        print(f"✗ Trainer integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running TrajectoryAware GRPO Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_trajectory_aware_grpo_loss,
        test_configuration,
        test_tensor_operations,
        test_trainer_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Trajectory-aware GRPO implementation is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)