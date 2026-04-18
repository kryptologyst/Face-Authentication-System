"""Tests for face authentication system."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.face_models import FaceNet, ArcFaceLoss, TripletLoss, LivenessDetector
from src.utils.core import set_seed, get_device, ConfigManager
from src.eval.metrics import BiometricEvaluator


class TestFaceModels:
    """Test face recognition models."""
    
    def setup_method(self):
        """Setup test method."""
        set_seed(42)
        self.device = get_device()
    
    def test_facenet_forward(self):
        """Test FaceNet forward pass."""
        model = FaceNet(embedding_dim=512).to(self.device)
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Forward pass
        embeddings = model(input_tensor)
        
        # Check output shape
        assert embeddings.shape == (batch_size, 512)
        
        # Check that embeddings are normalized
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_arcface_loss(self):
        """Test ArcFace loss computation."""
        embedding_dim = 512
        num_classes = 10
        batch_size = 4
        
        loss_fn = ArcFaceLoss(embedding_dim, num_classes)
        
        # Create dummy embeddings and labels
        embeddings = torch.randn(batch_size, embedding_dim)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Compute loss
        loss = loss_fn(embeddings, labels)
        
        # Check that loss is a scalar tensor
        assert loss.dim() == 0
        assert loss.item() > 0
    
    def test_triplet_loss(self):
        """Test triplet loss computation."""
        loss_fn = TripletLoss(margin=0.5)
        
        # Create dummy embeddings
        anchor = torch.randn(4, 512)
        positive = torch.randn(4, 512)
        negative = torch.randn(4, 512)
        
        # Compute loss
        loss = loss_fn(anchor, positive, negative)
        
        # Check that loss is a scalar tensor
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_liveness_detector(self):
        """Test liveness detector."""
        model = LivenessDetector().to(self.device)
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(self.device)
        
        # Forward pass
        output = model(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, 2)


class TestBiometricEvaluator:
    """Test biometric evaluation metrics."""
    
    def setup_method(self):
        """Setup test method."""
        set_seed(42)
        self.evaluator = BiometricEvaluator()
    
    def test_eer_computation(self):
        """Test EER computation."""
        # Create dummy scores
        genuine_scores = np.random.normal(0.8, 0.1, 100)
        impostor_scores = np.random.normal(0.3, 0.1, 100)
        
        eer, threshold = self.evaluator.compute_eer(genuine_scores, impostor_scores)
        
        # Check that EER is between 0 and 1
        assert 0 <= eer <= 1
        assert isinstance(threshold, float)
    
    def test_min_dcf_computation(self):
        """Test minDCF computation."""
        # Create dummy scores
        genuine_scores = np.random.normal(0.8, 0.1, 100)
        impostor_scores = np.random.normal(0.3, 0.1, 100)
        
        min_dcf, threshold = self.evaluator.compute_min_dcf(
            genuine_scores, impostor_scores
        )
        
        # Check that minDCF is non-negative
        assert min_dcf >= 0
        assert isinstance(threshold, float)
    
    def test_far_frr_computation(self):
        """Test FAR and FRR computation."""
        # Create dummy scores
        genuine_scores = np.random.normal(0.8, 0.1, 100)
        impostor_scores = np.random.normal(0.3, 0.1, 100)
        threshold = 0.5
        
        far, frr = self.evaluator.compute_far_frr(
            genuine_scores, impostor_scores, threshold
        )
        
        # Check that FAR and FRR are between 0 and 1
        assert 0 <= far <= 1
        assert 0 <= frr <= 1
    
    def test_roc_curve_computation(self):
        """Test ROC curve computation."""
        # Create dummy scores
        genuine_scores = np.random.normal(0.8, 0.1, 100)
        impostor_scores = np.random.normal(0.3, 0.1, 100)
        
        fpr, tpr, thresholds = self.evaluator.compute_roc_curve(
            genuine_scores, impostor_scores
        )
        
        # Check output shapes
        assert len(fpr) == len(tpr) == len(thresholds)
        assert len(fpr) > 0
        
        # Check that FPR and TPR are between 0 and 1
        assert np.all((fpr >= 0) & (fpr <= 1))
        assert np.all((tpr >= 0) & (tpr <= 1))


class TestConfigManager:
    """Test configuration manager."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        
        # Test getting non-existent key
        value = config_manager.get('non_existent_key', 'default')
        assert value == 'default'
    
    def test_config_manager_update(self):
        """Test ConfigManager update functionality."""
        config_manager = ConfigManager()
        
        # Update a value
        config_manager.update('test_key', 'test_value')
        
        # Get the value
        value = config_manager.get('test_key')
        assert value == 'test_value'


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        # This is a basic test - in practice, you'd test reproducibility
        set_seed(42)
        
        # Generate some random numbers
        rand1 = np.random.random()
        rand2 = torch.rand(1).item()
        
        # Set seed again and generate same numbers
        set_seed(42)
        rand1_again = np.random.random()
        rand2_again = torch.rand(1).item()
        
        # They should be the same
        assert rand1 == rand1_again
        assert rand2 == rand2_again
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        # Should return a torch device
        assert isinstance(device, torch.device)
        
        # Should be one of the expected devices
        assert device.type in ['cpu', 'cuda', 'mps']


if __name__ == "__main__":
    pytest.main([__file__])
