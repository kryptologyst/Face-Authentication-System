"""Evaluation metrics for biometric authentication systems."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve, 
    auc, 
    confusion_matrix,
    classification_report
)


class BiometricEvaluator:
    """Evaluator for biometric authentication systems."""
    
    def __init__(self):
        """Initialize biometric evaluator."""
        self.results = {}
    
    def compute_eer(
        self, 
        genuine_scores: np.ndarray, 
        impostor_scores: np.ndarray
    ) -> Tuple[float, float]:
        """Compute Equal Error Rate (EER).
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            
        Returns:
            Tuple of (EER, threshold_at_EER)
        """
        # Combine scores and labels
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Find EER (where FPR = 1 - TPR)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = fpr[eer_idx]
        eer_threshold = thresholds[eer_idx]
        
        return eer, eer_threshold
    
    def compute_min_dcf(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
    ) -> Tuple[float, float]:
        """Compute minimum Detection Cost Function (minDCF).
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            p_target: Target probability of false alarm
            c_miss: Cost of miss (false rejection)
            c_fa: Cost of false alarm (false acceptance)
            
        Returns:
            Tuple of (minDCF, threshold_at_minDCF)
        """
        # Combine scores and labels
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Compute DCF for each threshold
        fnr = 1 - tpr
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        
        # Find minimum DCF
        min_dcf_idx = np.argmin(dcf)
        min_dcf = dcf[min_dcf_idx]
        min_dcf_threshold = thresholds[min_dcf_idx]
        
        return min_dcf, min_dcf_threshold
    
    def compute_far_frr(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        threshold: float
    ) -> Tuple[float, float]:
        """Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR).
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            threshold: Decision threshold
            
        Returns:
            Tuple of (FAR, FRR)
        """
        # FAR: proportion of impostors accepted
        far = np.mean(impostor_scores >= threshold)
        
        # FRR: proportion of genuines rejected
        frr = np.mean(genuine_scores < threshold)
        
        return far, frr
    
    def compute_roc_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            
        Returns:
            Tuple of (FPR, TPR, thresholds)
        """
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        return fpr, tpr, thresholds
    
    def compute_det_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute DET curve (Detection Error Tradeoff).
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            
        Returns:
            Tuple of (FAR, FRR)
        """
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        # Convert to FAR and FRR
        far = fpr
        frr = 1 - tpr
        
        return far, frr
    
    def evaluate_authentication(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        p_target: float = 0.01,
        c_miss: float = 1.0,
        c_fa: float = 1.0
    ) -> Dict[str, float]:
        """Comprehensive authentication evaluation.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            p_target: Target probability for minDCF
            c_miss: Cost of miss
            c_fa: Cost of false alarm
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Basic metrics
        eer, eer_threshold = self.compute_eer(genuine_scores, impostor_scores)
        min_dcf, min_dcf_threshold = self.compute_min_dcf(
            genuine_scores, impostor_scores, p_target, c_miss, c_fa
        )
        
        # FAR and FRR at EER threshold
        far_eer, frr_eer = self.compute_far_frr(
            genuine_scores, impostor_scores, eer_threshold
        )
        
        # FAR and FRR at minDCF threshold
        far_min_dcf, frr_min_dcf = self.compute_far_frr(
            genuine_scores, impostor_scores, min_dcf_threshold
        )
        
        # ROC AUC
        fpr, tpr, _ = self.compute_roc_curve(genuine_scores, impostor_scores)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall AUC
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        results = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'min_dcf': min_dcf,
            'min_dcf_threshold': min_dcf_threshold,
            'far_at_eer': far_eer,
            'frr_at_eer': frr_eer,
            'far_at_min_dcf': far_min_dcf,
            'frr_at_min_dcf': frr_min_dcf,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'num_genuine': len(genuine_scores),
            'num_impostor': len(impostor_scores)
        }
        
        return results
    
    def evaluate_liveness_detection(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate liveness detection performance.
        
        Args:
            predictions: Liveness predictions (probabilities)
            labels: True labels (0: fake, 1: real)
            threshold: Decision threshold
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert predictions to binary
        binary_predictions = (predictions >= threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(labels, binary_predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Compute ROC metrics
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Compute precision-recall metrics
        precision_pr, recall_pr, _ = precision_recall_curve(labels, predictions)
        pr_auc = auc(recall_pr, precision_pr)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return results
    
    def plot_roc_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            title: Plot title
            save_path: Path to save plot
        """
        fpr, tpr, _ = self.compute_roc_curve(genuine_scores, impostor_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_det_curve(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        title: str = "DET Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot DET curve.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            title: Plot title
            save_path: Path to save plot
        """
        far, frr = self.compute_det_curve(genuine_scores, impostor_scores)
        
        plt.figure(figsize=(8, 6))
        plt.semilogx(far, frr, 'b-', lw=2, label='DET curve')
        plt.xlabel('False Acceptance Rate')
        plt.ylabel('False Rejection Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_score_distributions(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        title: str = "Score Distributions",
        save_path: Optional[str] = None
    ) -> None:
        """Plot score distributions.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            title: Plot title
            save_path: Path to save plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine', color='green')
        plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor', color='red')
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        liveness_predictions: Optional[np.ndarray] = None,
        liveness_labels: Optional[np.ndarray] = None
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            genuine_scores: Scores for genuine comparisons
            impostor_scores: Scores for impostor comparisons
            liveness_predictions: Liveness detection predictions
            liveness_labels: Liveness detection labels
            
        Returns:
            Formatted evaluation report
        """
        # Authentication evaluation
        auth_results = self.evaluate_authentication(genuine_scores, impostor_scores)
        
        report = "=" * 60 + "\n"
        report += "FACE AUTHENTICATION SYSTEM EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Authentication metrics
        report += "AUTHENTICATION PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        report += f"Equal Error Rate (EER): {auth_results['eer']:.4f}\n"
        report += f"EER Threshold: {auth_results['eer_threshold']:.4f}\n"
        report += f"Minimum DCF: {auth_results['min_dcf']:.4f}\n"
        report += f"MinDCF Threshold: {auth_results['min_dcf_threshold']:.4f}\n"
        report += f"FAR at EER: {auth_results['far_at_eer']:.4f}\n"
        report += f"FRR at EER: {auth_results['frr_at_eer']:.4f}\n"
        report += f"ROC AUC: {auth_results['roc_auc']:.4f}\n"
        report += f"PR AUC: {auth_results['pr_auc']:.4f}\n"
        report += f"Number of Genuine Comparisons: {auth_results['num_genuine']}\n"
        report += f"Number of Impostor Comparisons: {auth_results['num_impostor']}\n\n"
        
        # Liveness detection evaluation
        if liveness_predictions is not None and liveness_labels is not None:
            liveness_results = self.evaluate_liveness_detection(
                liveness_predictions, liveness_labels
            )
            
            report += "LIVENESS DETECTION PERFORMANCE:\n"
            report += "-" * 30 + "\n"
            report += f"Accuracy: {liveness_results['accuracy']:.4f}\n"
            report += f"Precision: {liveness_results['precision']:.4f}\n"
            report += f"Recall: {liveness_results['recall']:.4f}\n"
            report += f"Specificity: {liveness_results['specificity']:.4f}\n"
            report += f"F1 Score: {liveness_results['f1_score']:.4f}\n"
            report += f"ROC AUC: {liveness_results['roc_auc']:.4f}\n"
            report += f"PR AUC: {liveness_results['pr_auc']:.4f}\n"
            report += f"True Positives: {liveness_results['true_positives']}\n"
            report += f"True Negatives: {liveness_results['true_negatives']}\n"
            report += f"False Positives: {liveness_results['false_positives']}\n"
            report += f"False Negatives: {liveness_results['false_negatives']}\n\n"
        
        report += "=" * 60 + "\n"
        
        return report
