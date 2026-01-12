"""
L2 Linear Regression Pipeline for Polarization-based Solar Azimuth Prediction

This module implements the baseline L2 (Ridge) linear regression model for predicting 
solar azimuth from polarization data. According to the blueprint, this approach achieved 
2.375° accuracy in sub-optimal conditions in previous studies.

Key features:
- Ridge regularization to prevent overfitting
- Feature extraction from DoLP and AoLP data
- Cross-validation for robust evaluation
- Preprocessing pipeline for polarization images
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime


class L2PolarizationRegressor:
    """
    L2 (Ridge) Linear Regression model for solar azimuth prediction from polarization data.
    
    This serves as the baseline model according to the project blueprint, targeting
    < 5° Mean Absolute Error for solar azimuth prediction.
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 polynomial_degree: int = 1,
                 include_interactions: bool = True,
                 random_state: int = 42):
        """
        Initialize the L2 regression pipeline.
        
        Args:
            alpha: Ridge regularization strength (higher = more regularization)
            polynomial_degree: Degree for polynomial features
            include_interactions: Whether to include interaction terms
            random_state: Random seed for reproducibility
        """
        self.alpha = alpha
        self.polynomial_degree = polynomial_degree
        self.include_interactions = include_interactions
        self.random_state = random_state
        
        # Initialize components
        self.scaler = StandardScaler()
        self.poly_features = None
        self.regressor = Ridge(alpha=self.alpha, random_state=self.random_state)
        
        # Training history
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        
        if polynomial_degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=polynomial_degree, 
                include_bias=False,
                interaction_only=not include_interactions
            )
    
    def extract_polarization_features(self, dolp: np.ndarray, aolp: np.ndarray) -> np.ndarray:
        """
        Extract features from DoLP (Degree of Linear Polarization) and 
        AoLP (Angle of Linear Polarization) data.
        
        Args:
            dolp: Degree of Linear Polarization array, shape (N, H, W) or (N, pixels)
            aolp: Angle of Linear Polarization array, shape (N, H, W) or (N, pixels)
            
        Returns:
            Feature matrix of shape (N, num_features)
        """
        # Flatten spatial dimensions if needed
        if dolp.ndim == 3:
            dolp = dolp.reshape(dolp.shape[0], -1)
        if aolp.ndim == 3:
            aolp = aolp.reshape(aolp.shape[0], -1)
            
        features_list = []
        
        # Basic statistical features from DoLP
        features_list.extend([
            np.mean(dolp, axis=1),      # Mean DoLP
            np.std(dolp, axis=1),       # Std DoLP  
            np.median(dolp, axis=1),    # Median DoLP
            np.max(dolp, axis=1),       # Max DoLP
            np.min(dolp, axis=1),       # Min DoLP
        ])
        
        # Basic statistical features from AoLP
        features_list.extend([
            np.mean(aolp, axis=1),      # Mean AoLP
            np.std(aolp, axis=1),       # Std AoLP
            np.median(aolp, axis=1),    # Median AoLP
            np.max(aolp, axis=1),       # Max AoLP
            np.min(aolp, axis=1),       # Min AoLP
        ])
        
        # Trigonometric features from AoLP (important for circular data)
        aolp_rad = np.deg2rad(aolp)
        features_list.extend([
            np.mean(np.cos(2 * aolp_rad), axis=1),  # Mean cos(2*AoLP)
            np.mean(np.sin(2 * aolp_rad), axis=1),  # Mean sin(2*AoLP)
            np.std(np.cos(2 * aolp_rad), axis=1),   # Std cos(2*AoLP)
            np.std(np.sin(2 * aolp_rad), axis=1),   # Std sin(2*AoLP)
        ])
        
        # Combined features
        features_list.extend([
            np.mean(dolp * np.cos(2 * aolp_rad), axis=1),  # DoLP-weighted cos
            np.mean(dolp * np.sin(2 * aolp_rad), axis=1),  # DoLP-weighted sin
        ])
        
        # Stack all features
        features = np.column_stack(features_list)
        
        # Store feature names for interpretability
        if self.feature_names is None:
            self.feature_names = [
                'dolp_mean', 'dolp_std', 'dolp_median', 'dolp_max', 'dolp_min',
                'aolp_mean', 'aolp_std', 'aolp_median', 'aolp_max', 'aolp_min',
                'cos2_aolp_mean', 'sin2_aolp_mean', 'cos2_aolp_std', 'sin2_aolp_std',
                'dolp_cos2_mean', 'dolp_sin2_mean'
            ]
        
        return features
    
    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing transformations to features.
        
        Args:
            features: Raw feature matrix
            
        Returns:
            Preprocessed feature matrix
        """
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if self.is_fitted:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.fit_transform(features)
        
        # Apply polynomial features if specified
        if self.poly_features is not None:
            if self.is_fitted:
                features_scaled = self.poly_features.transform(features_scaled)
            else:
                features_scaled = self.poly_features.fit_transform(features_scaled)
        
        return features_scaled
    
    def fit(self, dolp: np.ndarray, aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """
        Train the L2 regression model.
        
        Args:
            dolp: Degree of Linear Polarization data
            aolp: Angle of Linear Polarization data  
            azimuth: Solar azimuth labels (target values)
            
        Returns:
            Dictionary containing training metrics
        """
        print("Extracting polarization features...")
        features = self.extract_polarization_features(dolp, aolp)
        
        print(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
        
        print("Preprocessing features...")
        X = self.prepare_features(features)
        
        print(f"Final feature matrix shape: {X.shape}")
        
        # Train the model
        print("Training Ridge regression model...")
        self.regressor.fit(X, azimuth)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.regressor.predict(X)
        
        # Convert errors from radians to degrees
        mae_rad = mean_absolute_error(azimuth, y_pred)
        rmse_rad = np.sqrt(mean_squared_error(azimuth, y_pred))
        
        metrics = {
            'mae': np.rad2deg(mae_rad),
            'rmse': np.rad2deg(rmse_rad),
            'r2': self.regressor.score(X, azimuth),
            'n_samples': len(azimuth),
            'n_features': X.shape[1]
        }
        
        self.training_metrics = metrics
        
        print(f"Training completed!")
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°") 
        print(f"Training R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, dolp: np.ndarray, aolp: np.ndarray) -> np.ndarray:
        """
        Predict solar azimuth from polarization data.
        
        Args:
            dolp: Degree of Linear Polarization data
            aolp: Angle of Linear Polarization data
            
        Returns:
            Predicted azimuth values in degrees
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        features = self.extract_polarization_features(dolp, aolp)
        X = self.prepare_features(features)
        
        return self.regressor.predict(X)
    
    def cross_validate(self, dolp: np.ndarray, aolp: np.ndarray, 
                      azimuth: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation as specified in the blueprint.
        
        Args:
            dolp: Degree of Linear Polarization data
            aolp: Angle of Linear Polarization data
            azimuth: Solar azimuth labels
            cv_folds: Number of cross-validation folds (default 5 per blueprint)
            
        Returns:
            Dictionary containing cross-validation metrics
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        features = self.extract_polarization_features(dolp, aolp)
        
        # Initialize cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        mae_scores = []
        rmse_scores = []
        r2_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            print(f"  Fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train, X_val = features[train_idx], features[val_idx]
            y_train, y_val = azimuth[train_idx], azimuth[val_idx]
            
            # Create a temporary model for this fold
            temp_scaler = StandardScaler()
            temp_poly = None
            if self.poly_features is not None:
                temp_poly = PolynomialFeatures(
                    degree=self.polynomial_degree,
                    include_bias=False,
                    interaction_only=not self.include_interactions
                )
            
            temp_regressor = Ridge(alpha=self.alpha, random_state=self.random_state)
            
            # Prepare training features
            X_train_scaled = temp_scaler.fit_transform(X_train)
            if temp_poly is not None:
                X_train_scaled = temp_poly.fit_transform(X_train_scaled)
            
            # Prepare validation features
            X_val_scaled = temp_scaler.transform(X_val)
            if temp_poly is not None:
                X_val_scaled = temp_poly.transform(X_val_scaled)
            
            # Train and predict
            temp_regressor.fit(X_train_scaled, y_train)
            y_pred = temp_regressor.predict(X_val_scaled)
            
            # Calculate metrics (convert from radians to degrees)
            mae = np.rad2deg(mean_absolute_error(y_val, y_pred))
            rmse = np.rad2deg(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2 = temp_regressor.score(X_val_scaled, y_val)
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            
            print(f"    MAE: {mae:.3f}°, RMSE: {rmse:.3f}°, R²: {r2:.3f}")
        
        cv_results = {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'mae_scores': mae_scores,
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores
        }
        
        print(f"\nCross-validation Results:")
        print(f"MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}°")
        print(f"RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}°")
        print(f"R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        
        # Check if meets blueprint requirements (MAE < 5°, RMSE ≤ 10%)
        meets_requirements = (cv_results['mae_mean'] < 5.0 and 
                            cv_results['rmse_mean'] <= 10.0)  # Assuming 10° as 10% of 100° range
        
        print(f"Meets blueprint requirements: {'✓' if meets_requirements else '✗'}")
        
        return cv_results
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        model_data = {
            'regressor': self.regressor,
            'scaler': self.scaler,
            'poly_features': self.poly_features,
            'hyperparameters': {
                'alpha': self.alpha,
                'polynomial_degree': self.polynomial_degree,
                'include_interactions': self.include_interactions,
                'random_state': self.random_state
            },
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'L2PolarizationRegressor':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(**model_data['hyperparameters'])
        
        # Restore fitted components
        instance.regressor = model_data['regressor']
        instance.scaler = model_data['scaler']
        instance.poly_features = model_data['poly_features']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.is_fitted = True
        
        print(f"Model loaded from {filepath}")
        return instance


def create_baseline_model(alpha: float = 1.0) -> L2PolarizationRegressor:
    """
    Create the baseline L2 regression model as specified in the blueprint.
    
    Args:
        alpha: Ridge regularization parameter
        
    Returns:
        Configured L2PolarizationRegressor instance
    """
    return L2PolarizationRegressor(
        alpha=alpha,
        polynomial_degree=1,  # Start with linear features
        include_interactions=False,
        random_state=42  # For reproducibility per blueprint
    )


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing L2 Polarization Regressor with mock data...")
    
    # Create mock data (replace with real data loading)
    n_samples = 1000
    h, w = 64, 64
    
    # Mock DoLP and AoLP data
    np.random.seed(42)
    dolp_mock = np.random.uniform(0, 1, (n_samples, h, w))
    aolp_mock = np.random.uniform(0, 180, (n_samples, h, w))
    
    # Mock azimuth labels (target variable)
    azimuth_mock = np.random.uniform(0, 360, n_samples)
    
    # Create and test model
    model = create_baseline_model(alpha=1.0)
    
    # Train model
    train_metrics = model.fit(dolp_mock, aolp_mock, azimuth_mock)
    
    # Perform cross-validation
    cv_metrics = model.cross_validate(dolp_mock, aolp_mock, azimuth_mock, cv_folds=5)
    
    print("\nBaseline L2 model testing completed!")
