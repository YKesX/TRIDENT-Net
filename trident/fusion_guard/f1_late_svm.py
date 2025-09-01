"""
TRIDENT-F1: Late Fusion SVM (scikit-learn wrapper)

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Union, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

from ..common.types import FusionModule, FeatureVec, OutcomeEstimate, EventToken


class LateFusionSVM(FusionModule):
    """
    Late Fusion SVM for multimodal feature combination.
    
    Uses scikit-learn SVM to combine features from multiple sensor branches
    in a classical machine learning approach.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        c: float = 1.0,
        probability: bool = True,
        random_state: int = 42,
    ):
        super().__init__(out_dim=1)  # SVM outputs single probability
        
        self.kernel = kernel
        self.gamma = gamma
        self.c = c
        self.probability = probability
        self.random_state = random_state
        
        # SVM model and scaler
        self.svm = SVC(
            kernel=kernel,
            gamma=gamma,
            C=c,
            probability=probability,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        
        # Training state
        self.is_fitted = False
        self.feature_dim = None
        
        # For storing training data
        self.training_features = []
        self.training_labels = []
    
    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: Optional[np.ndarray] = None,
        validation_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Fit the SVM model.
        
        Args:
            features: Training features of shape (N, D)
            labels: Training labels of shape (N,)
            validation_features: Optional validation features
            validation_labels: Optional validation labels
            
        Returns:
            Training metrics dictionary
        """
        # Fit scaler and transform features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit SVM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.svm.fit(features_scaled, labels)
        
        self.is_fitted = True
        self.feature_dim = features.shape[1]
        
        # Compute training metrics
        train_pred = self.svm.predict(features_scaled)
        train_acc = accuracy_score(labels, train_pred)
        
        metrics = {"train_accuracy": train_acc}
        
        # Validation metrics if provided
        if validation_features is not None and validation_labels is not None:
            val_features_scaled = self.scaler.transform(validation_features)
            val_pred = self.svm.predict(val_features_scaled)
            val_acc = accuracy_score(validation_labels, val_pred)
            metrics["val_accuracy"] = val_acc
        
        return metrics
    
    def forward(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None,
        z_t: Optional[FeatureVec] = None,
        events: Optional[List[EventToken]] = None,
    ) -> OutcomeEstimate:
        """
        Forward pass for SVM fusion.
        
        Args:
            z_r: Radar features
            z_i: Visible/EO features
            z_t: Thermal/IR features
            events: Event tokens (not used by SVM)
            
        Returns:
            OutcomeEstimate with SVM predictions
        """
        if not self.is_fitted:
            raise RuntimeError("SVM model not fitted. Call fit() first.")
        
        # Collect available features
        feature_list = []
        modality_info = []
        
        if z_r is not None:
            feature_list.append(z_r.z.detach().cpu().numpy())
            modality_info.append("radar")
        
        if z_i is not None:
            feature_list.append(z_i.z.detach().cpu().numpy())
            modality_info.append("visible")
        
        if z_t is not None:
            feature_list.append(z_t.z.detach().cpu().numpy())
            modality_info.append("thermal")
        
        if not feature_list:
            raise ValueError("At least one feature vector must be provided")
        
        # Concatenate features
        combined_features = np.concatenate(feature_list, axis=1)  # (B, total_dim)
        
        # Check feature dimension consistency
        if combined_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.feature_dim}, "
                f"got {combined_features.shape[1]}"
            )
        
        # Scale features
        features_scaled = self.scaler.transform(combined_features)
        
        # Predict probabilities
        if self.probability:
            proba = self.svm.predict_proba(features_scaled)
            # Assuming binary classification, take positive class probability
            p_outcome = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        else:
            # Use decision function as proxy for probability
            decision = self.svm.decision_function(features_scaled)
            p_outcome = 1 / (1 + np.exp(-decision))  # Sigmoid transformation
        
        # Convert to tensors
        device = feature_list[0].device if hasattr(feature_list[0], 'device') else torch.device('cpu')
        p_outcome_tensor = torch.tensor(p_outcome, dtype=torch.float32, device=device)
        binary_outcome = (p_outcome_tensor > 0.5).long()
        
        # Create explanation
        explanation = {
            "fusion_type": "late_svm",
            "modalities_used": modality_info,
            "feature_dimensions": [f.shape[1] for f in feature_list],
            "svm_params": {
                "kernel": self.kernel,
                "gamma": self.gamma,
                "C": self.c,
            },
            "decision_values": self.svm.decision_function(features_scaled).tolist(),
        }
        
        return OutcomeEstimate(
            p_outcome=p_outcome_tensor,
            binary_outcome=binary_outcome,
            explanation=explanation,
        )
    
    def add_training_sample(
        self,
        z_r: Optional[FeatureVec] = None,
        z_i: Optional[FeatureVec] = None,
        z_t: Optional[FeatureVec] = None,
        label: int = 0,
    ) -> None:
        """
        Add a training sample to the buffer.
        
        Args:
            z_r: Radar features
            z_i: Visible features  
            z_t: Thermal features
            label: Ground truth label
        """
        # Collect features
        feature_list = []
        
        if z_r is not None:
            feature_list.append(z_r.z.detach().cpu().numpy())
        if z_i is not None:
            feature_list.append(z_i.z.detach().cpu().numpy())
        if z_t is not None:
            feature_list.append(z_t.z.detach().cpu().numpy())
        
        if feature_list:
            combined_features = np.concatenate(feature_list, axis=1)
            # Handle batch dimension
            if len(combined_features.shape) == 2:
                for i in range(combined_features.shape[0]):
                    self.training_features.append(combined_features[i])
                    self.training_labels.append(label)
            else:
                self.training_features.append(combined_features)
                self.training_labels.append(label)
    
    def fit_from_buffer(self) -> Dict[str, float]:
        """Fit SVM using accumulated training samples."""
        if not self.training_features:
            raise ValueError("No training samples in buffer")
        
        features = np.array(self.training_features)
        labels = np.array(self.training_labels)
        
        return self.fit(features, labels)
    
    def save_model(self, path: str) -> None:
        """Save the fitted SVM model."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        model_data = {
            "svm": self.svm,
            "scaler": self.scaler,
            "feature_dim": self.feature_dim,
            "config": {
                "kernel": self.kernel,
                "gamma": self.gamma,
                "c": self.c,
                "probability": self.probability,
                "random_state": self.random_state,
            }
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> None:
        """Load a fitted SVM model."""
        model_data = joblib.load(path)
        
        self.svm = model_data["svm"]
        self.scaler = model_data["scaler"]
        self.feature_dim = model_data["feature_dim"]
        
        # Update config if available
        if "config" in model_data:
            config = model_data["config"]
            self.kernel = config["kernel"]
            self.gamma = config["gamma"]
            self.c = config["c"]
            self.probability = config["probability"]
            self.random_state = config["random_state"]
        
        self.is_fitted = True
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (for linear SVM only).
        
        Returns:
            Feature weights for linear kernel, None otherwise
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        if self.kernel == "linear":
            return self.svm.coef_[0]
        else:
            return None
    
    def get_support_vectors(self) -> Dict[str, np.ndarray]:
        """Get support vector information."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        return {
            "support_vectors": self.svm.support_vectors_,
            "support_indices": self.svm.support_,
            "n_support": self.svm.n_support_,
        }


def create_late_fusion_svm(config: dict) -> LateFusionSVM:
    """Factory function to create LateFusionSVM from config."""
    return LateFusionSVM(
        kernel=config.get("kernel", "rbf"),
        gamma=config.get("gamma", "scale"),
        c=config.get("c", 1.0),
        probability=config.get("probability", True),
        random_state=config.get("random_state", 42),
    )