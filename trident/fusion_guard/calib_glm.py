"""
CalibGLM - Classical GLM baseline for calibration checks

Author: Yağızhan Keskin
"""

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback implementations
    class LogisticRegression:
        def __init__(self, **kwargs):
            self.coef_ = [[1.0]]
            self.intercept_ = [0.0]
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])
        def predict_proba(self, X):
            return np.column_stack([np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5])
    
    class GradientBoostingClassifier:
        def __init__(self, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(X.shape[0])
        def predict_proba(self, X):
            return np.column_stack([np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5])
    
    class StandardScaler:
        def fit(self, X):
            return self
        def transform(self, X):
            return X
        def fit_transform(self, X):
            return X
    
    class joblib:
        @staticmethod
        def dump(obj, filename):
            pass
        @staticmethod
        def load(filename):
            return LogisticRegression()
    
    SKLEARN_AVAILABLE = False

from ..common.types import FusionModule, EventToken


class CalibGLM(FusionModule):
    """
    Logistic or GBM baseline on detached features for sanity and calibration checks.
    
    Classical machine learning baseline using scikit-learn for
    outcome prediction from concatenated multimodal features.
    
    Input: features (B, 1696) - concat(zi, zt, zr, e_cls) => 768+512+384+32
    Outputs:
        - p_hit_aux (B, 1) - auxiliary hit probability  
        - p_kill_aux (B, 1) - auxiliary kill probability
    """
    
    def __init__(self, in_dim: int = 1696, model: str = "logreg", c: float = 1.0, 
                 max_iter: int = 200, random_state: int = 42):
        super().__init__(out_dim=2)  # Two outputs: p_hit_aux, p_kill_aux
        
        self.in_dim = in_dim
        self.model_type = model
        self.c = c
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Models for hit and kill prediction
        self.hit_model = None
        self.kill_model = None
        
        # Initialize models
        self._init_models()
        
        # Training state
        self.is_fitted = False
        
    def _init_models(self):
        """Initialize the classical ML models."""
        if self.model_type == "logreg":
            # Logistic regression
            self.hit_model = LogisticRegression(
                C=self.c,
                random_state=self.random_state,
                max_iter=self.max_iter,
                solver='liblinear'
            )
            self.kill_model = LogisticRegression(
                C=self.c,
                random_state=self.random_state,
                max_iter=self.max_iter,
                solver='liblinear'
            )
        elif self.model_type == "xgboost":
            # Gradient boosting
            self.hit_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
            self.kill_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, features: torch.Tensor, hit_labels: torch.Tensor, kill_labels: torch.Tensor):
        """
        Fit the classical models.
        
        Args:
            features: Input features (N, 1664)
            hit_labels: Hit labels (N,)
            kill_labels: Kill labels (N,)
        """
        # Convert to numpy
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(hit_labels, torch.Tensor):
            hit_labels = hit_labels.detach().cpu().numpy()
        if isinstance(kill_labels, torch.Tensor):
            kill_labels = kill_labels.detach().cpu().numpy()
        
        # Fit scaler
        self.scaler.fit(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Fit models
        self.hit_model.fit(features_scaled, hit_labels)
        self.kill_model.fit(features_scaled, kill_labels)
        
        self.is_fitted = True
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CalibGLM.
        
        Args:
            features: Concatenated features (B, 1664)
            
        Returns:
            tuple: (p_hit_aux, p_kill_aux)
                - p_hit_aux: (B, 1) auxiliary hit probabilities
                - p_kill_aux: (B, 1) auxiliary kill probabilities
        """
        if not self.is_fitted:
            # Return dummy predictions if not fitted
            B = features.shape[0]
            device = features.device
            return (torch.full((B, 1), 0.5, device=device),
                   torch.full((B, 1), 0.5, device=device))
        
        # Convert to numpy for sklearn
        features_np = features.detach().cpu().numpy()
        
        # Scale features
        features_scaled = self.scaler.transform(features_np)
        
        # Predict probabilities
        hit_probs = self.hit_model.predict_proba(features_scaled)[:, 1]  # Get positive class prob
        kill_probs = self.kill_model.predict_proba(features_scaled)[:, 1]
        
        # Convert back to torch tensors
        device = features.device
        p_hit_aux = torch.from_numpy(hit_probs).float().unsqueeze(1).to(device)
        p_kill_aux = torch.from_numpy(kill_probs).float().unsqueeze(1).to(device)
        
        return p_hit_aux, p_kill_aux
    
    def save(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        save_data = {
            'hit_model': self.hit_model,
            'kill_model': self.kill_model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'c': self.c,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
    
    def load(self, filepath: str):
        """Load a trained model."""
        save_data = joblib.load(filepath)
        
        self.hit_model = save_data['hit_model']
        self.kill_model = save_data['kill_model']
        self.scaler = save_data['scaler']
        self.model_type = save_data['model_type']
        self.c = save_data.get('c', 1.0)
        self.max_iter = save_data.get('max_iter', 200)
        self.random_state = save_data.get('random_state', 42)
        self.is_fitted = save_data['is_fitted']
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from trained models."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to get feature importance")
        
        importance = {}
        
        if self.model_type == "logreg":
            # Use coefficients as importance
            importance['hit'] = np.abs(self.hit_model.coef_[0])
            importance['kill'] = np.abs(self.kill_model.coef_[0])
        elif self.model_type == "xgboost":
            # Use feature importance
            importance['hit'] = self.hit_model.feature_importances_
            importance['kill'] = self.kill_model.feature_importances_
        
        return importance
    
    def predict_with_uncertainty(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimates.
        
        Returns:
            tuple: (p_hit_aux, p_kill_aux, hit_uncertainty, kill_uncertainty)
        """
        p_hit_aux, p_kill_aux = self.forward(features)
        
        # Simple uncertainty based on distance from 0.5
        hit_uncertainty = 2 * torch.abs(p_hit_aux - 0.5)  # Higher when closer to decision boundary
        kill_uncertainty = 2 * torch.abs(p_kill_aux - 0.5)
        
        return p_hit_aux, p_kill_aux, hit_uncertainty, kill_uncertainty