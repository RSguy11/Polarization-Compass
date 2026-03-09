"""
Support Vector Regression for Azimuth Prediction
===============================================

SVR implementation that treats azimuth as a continuous regression problem
instead of discretizing into bins. Handles circular nature of angles.
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class CircularSVR:
    """
    Support Vector Regression for circular azimuth prediction.
    
    Converts circular degrees to complex representation for better regression,
    then converts back to degrees.
    """
    
    def __init__(self, C=1.0, gamma='scale', kernel='rbf', epsilon=0.1, 
                 feature_selection=None, use_complex=True):
        """
        Initialize CircularSVR.
        
        Args:
            C: Regularization parameter
            gamma: Kernel coefficient
            kernel: Kernel type
            epsilon: Epsilon parameter for SVR
            feature_selection: Number of features to select (None = all)
            use_complex: Whether to use complex representation
        """
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.epsilon = epsilon
        self.feature_selection = feature_selection
        self.use_complex = use_complex
        
        # Initialize components
        self.scaler = StandardScaler()
        self.selector = None
        self.svr_real = None
        self.svr_imag = None
        self.svr_direct = None
        
    def _degrees_to_complex(self, degrees):
        """Convert degrees to complex representation."""
        radians = np.deg2rad(degrees)
        return np.cos(radians), np.sin(radians)
    
    def _complex_to_degrees(self, real, imag):
        """Convert complex representation back to degrees."""
        radians = np.arctan2(imag, real)
        degrees = np.rad2deg(radians) % 360
        return degrees
    
    def fit(self, X, y_degrees):
        """
        Train the SVR model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_degrees: Azimuth angles in degrees
        """
        print(f"Training CircularSVR (complex={self.use_complex})")
        
        # Feature scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selection is not None and self.feature_selection < X.shape[1]:
            self.selector = SelectKBest(f_regression, k=self.feature_selection)
            X_scaled = self.selector.fit_transform(X_scaled, y_degrees)
            print(f"Selected top {self.feature_selection} features from {X.shape[1]}")
        
        if self.use_complex:
            # Convert to complex representation
            y_real, y_imag = self._degrees_to_complex(y_degrees)
            
            # Train separate SVRs for real and imaginary parts
            self.svr_real = SVR(
                C=self.C, gamma=self.gamma, kernel=self.kernel, epsilon=self.epsilon
            )
            self.svr_imag = SVR(
                C=self.C, gamma=self.gamma, kernel=self.kernel, epsilon=self.epsilon
            )
            
            self.svr_real.fit(X_scaled, y_real)
            self.svr_imag.fit(X_scaled, y_imag)
            
            # Training accuracy
            pred_real = self.svr_real.predict(X_scaled)
            pred_imag = self.svr_imag.predict(X_scaled)
            train_pred = self._complex_to_degrees(pred_real, pred_imag)
            
        else:
            # Direct regression on degrees
            self.svr_direct = SVR(
                C=self.C, gamma=self.gamma, kernel=self.kernel, epsilon=self.epsilon
            )
            self.svr_direct.fit(X_scaled, y_degrees)
            train_pred = self.svr_direct.predict(X_scaled)
        
        # Calculate training error
        train_error = self._circular_mae(y_degrees, train_pred)
        print(f"Training MAE: {train_error:.2f}°")
        
        return self
    
    def predict(self, X):
        """
        Predict azimuth angles.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted azimuth angles in degrees
        """
        # Apply same preprocessing
        X_scaled = self.scaler.transform(X)
        if self.selector is not None:
            X_scaled = self.selector.transform(X_scaled)
        
        if self.use_complex:
            # Predict real and imaginary parts
            pred_real = self.svr_real.predict(X_scaled)
            pred_imag = self.svr_imag.predict(X_scaled)
            
            # Convert back to degrees
            return self._complex_to_degrees(pred_real, pred_imag)
        else:
            # Direct prediction
            pred = self.svr_direct.predict(X_scaled)
            return pred % 360  # Ensure [0, 360) range
    
    def _circular_mae(self, y_true, y_pred):
        """Calculate circular Mean Absolute Error."""
        diff = np.abs(y_true - y_pred)
        diff = np.minimum(diff, 360 - diff)  # Handle wraparound
        return np.mean(diff)


class SVRWrapper:
    """
    Wrapper for SVR that matches the SVM interface for easy testing.
    """
    
    def __init__(self, C=1.0, gamma='scale', epsilon=0.1, feature_selection=100):
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.feature_selection = feature_selection
        self.svr = None
        
    def fit(self, X, y_radians):
        """Fit SVR (expects radians for interface compatibility)."""
        y_degrees = np.rad2deg(y_radians) % 360
        
        self.svr = CircularSVR(
            C=self.C,
            gamma=self.gamma, 
            epsilon=self.epsilon,
            feature_selection=self.feature_selection,
            use_complex=True
        )
        self.svr.fit(X, y_degrees)
        return self
        
    def predict(self, X):
        """Predict and return radians for interface compatibility."""
        degrees = self.svr.predict(X)
        return np.deg2rad(degrees)


def calculate_circular_mae_svr(y_true_deg, y_pred_deg):
    """Calculate circular MAE for SVR results."""
    diff = np.abs(y_true_deg - y_pred_deg)
    diff = np.minimum(diff, 360 - diff)
    mae = np.mean(diff)
    
    # Also calculate RMSE
    diff_rad = np.deg2rad(diff)
    rmse = np.sqrt(np.mean(diff_rad**2))
    rmse_deg = np.rad2deg(rmse)
    
    return mae, rmse_deg