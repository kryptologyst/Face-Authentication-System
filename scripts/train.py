#!/usr/bin/env python3
"""Training script for face authentication system."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.face_authentication import FaceAuthenticator
from src.data.datasets import create_data_loaders
from src.utils.core import setup_logging, set_seed, get_device, ConfigManager


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        labels = batch['user_id'].to(device)
        liveness_labels = batch['liveness_label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, labels)
        
        # Compute loss
        losses = model.compute_loss(outputs, labels, liveness_labels)
        loss = losses['total_loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return {'train_loss': avg_loss}


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch}"):
            images = batch['image'].to(device)
            labels = batch['user_id'].to(device)
            liveness_labels = batch['liveness_label'].to(device)
            
            # Forward pass
            outputs = model(images, labels)
            
            # Compute loss
            losses = model.compute_loss(outputs, labels, liveness_labels)
            loss = losses['total_loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return {'val_loss': avg_loss}


def train_model(
    config_path: str,
    data_dir: str,
    output_dir: str,
    resume_from: str = None
) -> None:
    """Train the face authentication model.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing training data
        output_dir: Directory to save model and logs
        resume_from: Path to checkpoint to resume from
    """
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting model training")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    
    # Set random seed
    seed = config_manager.get('reproducibility.seed', 42)
    set_seed(seed)
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Loading data...")
    data_loaders = create_data_loaders(
        data_dir=data_dir,
        batch_size=config_manager.get('training.batch_size', 32),
        num_workers=config_manager.get('training.num_workers', 4),
        image_size=config_manager.get('data.image_size', 224),
        augmentation=config_manager.get('data.augmentation', True)
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = FaceAuthenticator(config_path=config_path, device=device)
    
    # Setup optimizer
    learning_rate = config_manager.get('training.learning_rate', 0.001)
    weight_decay = config_manager.get('training.weight_decay', 1e-4)
    
    optimizer = optim.Adam(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Setup scheduler
    scheduler_type = config_manager.get('training.scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config_manager.get('training.epochs', 100)
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None
    
    # Setup loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup tensorboard
    writer = SummaryWriter(output_path / 'logs')
    
    # Training loop
    epochs = config_manager.get('training.epochs', 100)
    patience = config_manager.get('training.patience', 10)
    min_delta = config_manager.get('training.min_delta', 0.001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"Starting training for {epochs} epochs")
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model.model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch
        )
        
        # Validate
        val_metrics = validate_epoch(
            model.model,
            val_loader,
            criterion,
            device,
            epoch
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                   f"Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Tensorboard logging
        writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/Validation', val_metrics['val_loss'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Early stopping
        val_loss = val_metrics['val_loss']
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model.save_model(output_path / 'best_model.pth')
            logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save final model
    model.save_model(output_path / 'final_model.pth')
    
    # Close tensorboard writer
    writer.close()
    
    logger.info("Training completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train face authentication model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/trained",
        help="Directory to save model and logs"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    train_model(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
