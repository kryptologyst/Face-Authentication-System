#!/usr/bin/env python3
"""Evaluation script for face authentication system."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.face_authentication import FaceAuthenticator
from src.data.datasets import FaceDataset
from src.eval.metrics import BiometricEvaluator
from src.utils.core import setup_logging, ConfigManager


def evaluate_model(
    model_path: str,
    test_data_dir: str,
    output_dir: str,
    config_path: str = None
) -> Dict[str, Any]:
    """Evaluate the face authentication model.
    
    Args:
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        output_dir: Directory to save evaluation results
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing evaluation results
    """
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting model evaluation")
    
    # Load configuration
    config_manager = ConfigManager(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    authenticator = FaceAuthenticator(
        config_path=config_path,
        model_path=model_path
    )
    
    # Load test data
    logger.info("Loading test data...")
    test_dataset = FaceDataset(
        data_dir=test_data_dir,
        split="test",
        image_size=config_manager.get('data.image_size', 224),
        augmentation=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = BiometricEvaluator()
    
    # Collect scores
    logger.info("Collecting scores...")
    genuine_scores = []
    impostor_scores = []
    liveness_predictions = []
    liveness_labels = []
    
    for batch in test_loader:
        image = batch['image'].to(authenticator.device)
        user_id = batch['user_id'].item()
        user_name = batch['user_name'][0]
        is_enrollment = batch['is_enrollment'].item()
        liveness_label = batch['liveness_label'].item()
        
        # Skip enrollment images
        if is_enrollment:
            continue
        
        # Generate embeddings
        with torch.no_grad():
            outputs = authenticator.model(image)
            embedding = outputs['face_embeddings'].cpu().numpy()[0]
            
            # Liveness detection
            liveness_output = torch.nn.functional.softmax(outputs['liveness_output'], dim=1)
            liveness_pred = liveness_output[0, 1].cpu().numpy()
        
        # Check if user is enrolled
        if user_name in authenticator.enrolled_users:
            # Genuine comparison
            enrolled_embedding = authenticator.enrolled_users[user_name]['embedding']
            similarity = np.dot(embedding, enrolled_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(enrolled_embedding)
            )
            genuine_scores.append(similarity)
        else:
            # Impostor comparison (compare with random enrolled user)
            if authenticator.enrolled_users:
                random_user = list(authenticator.enrolled_users.keys())[0]
                enrolled_embedding = authenticator.enrolled_users[random_user]['embedding']
                similarity = np.dot(embedding, enrolled_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(enrolled_embedding)
                )
                impostor_scores.append(similarity)
        
        # Collect liveness data
        liveness_predictions.append(liveness_pred)
        liveness_labels.append(liveness_label)
    
    # Convert to numpy arrays
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    liveness_predictions = np.array(liveness_predictions)
    liveness_labels = np.array(liveness_labels)
    
    logger.info(f"Genuine comparisons: {len(genuine_scores)}")
    logger.info(f"Impostor comparisons: {len(impostor_scores)}")
    
    # Evaluate authentication
    logger.info("Evaluating authentication performance...")
    auth_results = evaluator.evaluate_authentication(
        genuine_scores,
        impostor_scores,
        p_target=config_manager.get('evaluation.min_dcf_p_target', 0.01),
        c_miss=config_manager.get('evaluation.min_dcf_c_miss', 1.0),
        c_fa=config_manager.get('evaluation.min_dcf_c_fa', 1.0)
    )
    
    # Evaluate liveness detection
    logger.info("Evaluating liveness detection performance...")
    liveness_results = evaluator.evaluate_liveness_detection(
        liveness_predictions,
        liveness_labels
    )
    
    # Combine results
    results = {
        'authentication': auth_results,
        'liveness_detection': liveness_results,
        'num_genuine': len(genuine_scores),
        'num_impostor': len(impostor_scores),
        'model_path': model_path,
        'test_data_dir': test_data_dir
    }
    
    # Generate report
    logger.info("Generating evaluation report...")
    report = evaluator.generate_report(
        genuine_scores,
        impostor_scores,
        liveness_predictions,
        liveness_labels
    )
    
    # Save results
    logger.info(f"Saving results to {output_path}")
    
    # Save numerical results
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save report
    with open(output_path / 'evaluation_report.txt', 'w') as f:
        f.write(report)
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    
    evaluator.plot_roc_curve(
        genuine_scores,
        impostor_scores,
        title="ROC Curve - Face Authentication",
        save_path=output_path / 'roc_curve.png'
    )
    
    evaluator.plot_det_curve(
        genuine_scores,
        impostor_scores,
        title="DET Curve - Face Authentication",
        save_path=output_path / 'det_curve.png'
    )
    
    evaluator.plot_score_distributions(
        genuine_scores,
        impostor_scores,
        title="Score Distributions - Face Authentication",
        save_path=output_path / 'score_distributions.png'
    )
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"EER: {auth_results['eer']:.4f}")
    logger.info(f"MinDCF: {auth_results['min_dcf']:.4f}")
    logger.info(f"ROC AUC: {auth_results['roc_auc']:.4f}")
    logger.info(f"Liveness Accuracy: {liveness_results['accuracy']:.4f}")
    logger.info(f"Liveness F1 Score: {liveness_results['f1_score']:.4f}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate face authentication model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default="data/test",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        config_path=args.config
    )
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
