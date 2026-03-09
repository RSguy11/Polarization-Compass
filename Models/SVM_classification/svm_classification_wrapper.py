"""
Support Vector Machine Classification for Azimuth Prediction
=============================================================

This module implements SVM classification approach for polarization compass azimuth prediction.
Converts continuous azimuth angles to discrete classes, trains SVM classifier, and converts back
to continuous angles for evaluation.

Key Features:
- Azimuth discretization with configurable bin sizes
- Circular data handling (359° → 1° continuity)
- Class probability prediction for uncertainty estimation
- Feature selection integration
- Cross-day domain adaptation support
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import warnings


class CircularSVMClassifier:
    """
    SVM classifier for circular azimuth data with bin-based discretization.
    
    Args:
        n_bins (int): Number of azimuth bins (8=45°, 16=22.5°, 32=11.25°)
        C (float): SVM regularization parameter
        gamma (str/float): RBF kernel parameter
        probability (bool): Enable probability estimation
        feature_selection (int): Number of top features to select (None=all)
        class_weight (str): Handle class imbalance ('balanced'/None)
    """
    
    def __init__(self, n_bins=16, C=1.0, gamma='scale', probability=True, 
                 feature_selection=100, class_weight='balanced', random_state=42):
        self.n_bins = n_bins
        self.bin_size = 360.0 / n_bins
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.feature_selection = feature_selection
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Initialize components
        self.scaler = StandardScaler()
        self.selector = None
        self.svm = None
        self.label_encoder = LabelEncoder()
        
        # Bin centers for angle reconstruction
        self.bin_centers = np.arange(self.bin_size/2, 360, self.bin_size)
        
    def _azimuth_to_bins(self, azimuth_deg):
        """Convert continuous azimuth angles to discrete bins."""
        # Ensure angles are in [0, 360) range
        azimuth_deg = azimuth_deg % 360
        
        # Assign to bins (center-aligned)
        bin_indices = np.floor(azimuth_deg / self.bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        return bin_indices
    
    def _bins_to_azimuth(self, bin_indices, probabilities=None):
        """
        Convert discrete bins back to continuous azimuth angles.
        
        Args:
            bin_indices: Predicted bin indices
            probabilities: Class probabilities (optional)
            
        Returns:
            Predicted azimuth angles in degrees
        """
        if probabilities is not None:
            # Probability-weighted prediction
            angles = []
            
            # Get the classes that the SVM was actually trained on
            trained_classes = self.svm.classes_
            
            for i, probs in enumerate(probabilities):
                # Create full probability array for all bins
                full_probs = np.zeros(self.n_bins)
                
                # Map trained class probabilities to corresponding bins
                for j, class_idx in enumerate(trained_classes):
                    if class_idx < self.n_bins:  # Safety check
                        full_probs[class_idx] = probs[j]
                
                # Calculate expected angle using circular statistics
                bin_angles_rad = np.deg2rad(self.bin_centers)
                
                # Convert to complex representation
                complex_angles = np.exp(1j * bin_angles_rad)
                weighted_complex = np.sum(full_probs * complex_angles)
                
                # Convert back to angle
                predicted_angle = np.rad2deg(np.angle(weighted_complex)) % 360
                angles.append(predicted_angle)
            
            return np.array(angles)
        else:
            # Simple bin center prediction
            return self.bin_centers[bin_indices]
    
    def fit(self, X, y_degrees):
        """
        Train the SVM classifier.
        
        Args:
            X (array): Feature matrix (n_samples, n_features)
            y_degrees (array): Azimuth angles in degrees
        """
        print(f"Training CircularSVMClassifier with {self.n_bins} bins ({self.bin_size:.1f}° resolution)")
        
        # Convert angles to bins
        y_bins = self._azimuth_to_bins(y_degrees)
        
        # Check class distribution
        unique_bins, counts = np.unique(y_bins, return_counts=True)
        print(f"Class distribution: {len(unique_bins)} classes, "
              f"min={counts.min()}, max={counts.max()}, avg={counts.mean():.1f}")
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selection is not None and self.feature_selection < X.shape[1]:
            self.selector = SelectKBest(f_classif, k=self.feature_selection)
            X_scaled = self.selector.fit_transform(X_scaled, y_bins)
            print(f"Selected top {self.feature_selection} features from {X.shape[1]}")
        
        # Train SVM classifier
        self.svm = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel='rbf',
            probability=self.probability,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        self.svm.fit(X_scaled, y_bins)
        
        # Training accuracy
        train_pred_bins = self.svm.predict(X_scaled)
        train_accuracy = np.mean(train_pred_bins == y_bins)
        print(f"Training bin accuracy: {train_accuracy:.3f}")
        
        return self
    
    def predict(self, X):
        """
        Predict azimuth angles.
        
        Args:
            X (array): Feature matrix
            
        Returns:
            Predicted azimuth angles in degrees
        """
        # Apply same preprocessing
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        
        # Get predictions
        if self.probability:
            probabilities = self.svm.predict_proba(X_scaled)
            # Use probability-weighted prediction
            return self._bins_to_azimuth(None, probabilities)
        else:
            pred_bins = self.svm.predict(X_scaled)
            return self._bins_to_azimuth(pred_bins)
    
    def predict_proba(self, X):
        """Get class probabilities."""
        if not self.probability:
            raise ValueError("Probability estimation not enabled")
        
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        
        return self.svm.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance based on SVM dual coefficients."""
        if self.svm is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.svm, 'dual_coef_'):
            # For multi-class SVM, approximate importance using dual coefficients
            support_vectors = self.svm.support_vectors_
            dual_coef = self.svm.dual_coef_
            
            # Calculate approximate feature importance
            importance = np.abs(dual_coef.T @ support_vectors).mean(axis=0)
            
            if self.selector is not None:
                # Map back to original features
                full_importance = np.zeros(self.selector.n_features_in_)
                selected_indices = self.selector.get_support(indices=True)
                full_importance[selected_indices] = importance
                return full_importance
            
            return importance
        
        return None


def calculate_circular_error(y_true_deg, y_pred_deg):
    """
    Calculate circular Mean Absolute Error for azimuth predictions.
    
    Args:
        y_true_deg, y_pred_deg: Arrays of angles in degrees
        
    Returns:
        mae_deg: Mean absolute error in degrees
        rmse_deg: Root mean square error in degrees
    """
    # Convert to radians for circular calculation
    true_rad = np.deg2rad(y_true_deg)
    pred_rad = np.deg2rad(y_pred_deg)
    
    # Calculate circular differences
    diff_rad = np.angle(np.exp(1j * (pred_rad - true_rad)))
    diff_deg = np.abs(np.rad2deg(diff_rad))
    
    mae_deg = np.mean(diff_deg)
    rmse_deg = np.sqrt(np.mean(diff_deg ** 2))
    
    return mae_deg, rmse_deg


class SVMClassificationWrapper:
    """
    Wrapper that mimics the interface of CircularRegressionWrapper
    for integration with existing pipeline.
    """
    
    def __init__(self, **svm_params):
        """Initialize with SVM parameters."""
        self.svm_params = svm_params
        self.model = None
        
    def fit(self, X, y_radians):
        """Fit the model with radians input (to match existing interface)."""
        y_degrees = np.rad2deg(y_radians) % 360
        
        self.model = CircularSVMClassifier(**self.svm_params)
        self.model.fit(X, y_degrees)
        
        return self
    
    def predict(self, X):
        """Predict and return in radians (to match existing interface)."""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        y_degrees = self.model.predict(X)
        return np.deg2rad(y_degrees)
    
    def score(self, X, y_radians):
        """Calculate negative MAE for sklearn compatibility."""
        y_pred_radians = self.predict(X)
        y_true_deg = np.rad2deg(y_radians) % 360
        y_pred_deg = np.rad2deg(y_pred_radians) % 360
        
        mae_deg, _ = calculate_circular_error(y_true_deg, y_pred_deg)
        return -mae_deg  # Negative for sklearn (higher is better)