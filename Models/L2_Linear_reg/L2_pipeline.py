"""L2 Linear Regression Pipeline for Polarization-based Solar Azimuth Prediction"""

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
    """L2 (Ridge) Linear Regression model for solar azimuth prediction from polarization data."""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 polynomial_degree: int = 1,
                 include_interactions: bool = True,
                 random_state: int = 42):
        """Initialize the L2 regression pipeline."""
        self.alpha = alpha
        self.polynomial_degree = polynomial_degree
        self.include_interactions = include_interactions
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.poly_features = None
        self.regressor = Ridge(alpha=self.alpha, random_state=self.random_state)
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
        """Extract 16 statistical features from DoLP and AoLP data."""
        if dolp.ndim == 3:
            dolp = dolp.reshape(dolp.shape[0], -1)
        if aolp.ndim == 3:
            aolp = aolp.reshape(aolp.shape[0], -1)
            
        features_list = [
            np.mean(dolp, axis=1), np.std(dolp, axis=1), np.median(dolp, axis=1),
            np.max(dolp, axis=1), np.min(dolp, axis=1),
            np.mean(aolp, axis=1), np.std(aolp, axis=1), np.median(aolp, axis=1),
            np.max(aolp, axis=1), np.min(aolp, axis=1)
        ]
        
        aolp_rad = np.deg2rad(aolp)
        features_list.extend([
            np.mean(np.cos(2 * aolp_rad), axis=1),
            np.mean(np.sin(2 * aolp_rad), axis=1),
            np.std(np.cos(2 * aolp_rad), axis=1),
            np.std(np.sin(2 * aolp_rad), axis=1),
            np.mean(dolp * np.cos(2 * aolp_rad), axis=1),
            np.mean(dolp * np.sin(2 * aolp_rad), axis=1)
        ])
        
        features = np.column_stack(features_list)
        
        if self.feature_names is None:
            self.feature_names = [
                'dolp_mean', 'dolp_std', 'dolp_median', 'dolp_max', 'dolp_min',
                'aolp_mean', 'aolp_std', 'aolp_median', 'aolp_max', 'aolp_min',
                'cos2_aolp_mean', 'sin2_aolp_mean', 'cos2_aolp_std', 'sin2_aolp_std',
                'dolp_cos2_mean', 'dolp_sin2_mean'
            ]
        
        return features
    
    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformations to features."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.is_fitted:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.fit_transform(features)
        
        if self.poly_features is not None:
            if self.is_fitted:
                features_scaled = self.poly_features.transform(features_scaled)
            else:
                features_scaled = self.poly_features.fit_transform(features_scaled)
        
        return features_scaled
    
    def fit_from_features(self, features: np.ndarray, azimuth: np.ndarray) -> Dict:
        """Train the L2 regression model using pre-extracted features."""
        print(f"Training with {features.shape[0]} samples, {features.shape[1]} features")
        print("Preprocessing features...")
        X = self.prepare_features(features)
        print(f"Final feature matrix shape: {X.shape}")
        print("Training Ridge regression model...")
        
        self.regressor.fit(X, azimuth)
        self.is_fitted = True
        
        y_pred = self.regressor.predict(X)
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
    
    def fit(self, dolp: np.ndarray, aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """Train the L2 regression model."""
        print("Extracting polarization features...")
        features = self.extract_polarization_features(dolp, aolp)
        print(f"Extracted {features.shape[1]} features from {features.shape[0]} samples")
        return self.fit_from_features(features, azimuth)
    
    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """Predict solar azimuth from pre-extracted features."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self.prepare_features(features)
        return self.regressor.predict(X)
    
    def predict(self, dolp: np.ndarray, aolp: np.ndarray) -> np.ndarray:
        """Predict solar azimuth from polarization data."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        features = self.extract_polarization_features(dolp, aolp)
        return self.predict_from_features(features)
    
    def cross_validate_from_features(self, features: np.ndarray, azimuth: np.ndarray, cv_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation using pre-extracted features."""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        mae_scores, rmse_scores, r2_scores = [], [], []
        
        print(f"Performing {cv_folds}-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
            temp_model = L2PolarizationRegressor(alpha=self.regressor.alpha)
            temp_model.fit_from_features(features[train_idx], azimuth[train_idx])
            y_pred = temp_model.predict_from_features(features[val_idx])
            
            mae = np.rad2deg(mean_absolute_error(azimuth[val_idx], y_pred))
            rmse = np.rad2deg(np.sqrt(mean_squared_error(azimuth[val_idx], y_pred)))
            r2 = temp_model.regressor.score(temp_model.prepare_features(features[val_idx]), azimuth[val_idx])
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            r2_scores.append(r2)
            print(f"  Fold {fold}/{cv_folds}")
            print(f"    MAE: {mae:.3f}°, RMSE: {rmse:.3f}°, R²: {r2:.3f}")
        
        results = {
            'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores), 'rmse_std': np.std(rmse_scores),
            'r2_mean': np.mean(r2_scores), 'r2_std': np.std(r2_scores)
        }
        
        print(f"\nCross-validation Results:")
        print(f"MAE: {results['mae_mean']:.3f} ± {results['mae_std']:.3f}°")
        print(f"RMSE: {results['rmse_mean']:.3f} ± {results['rmse_std']:.3f}°")
        print(f"R²: {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
        print(f"Meets blueprint requirements: {'✓' if results['mae_mean'] < 5.0 else '✗'}")
        
        return results
    
    def cross_validate(self, dolp: np.ndarray, aolp: np.ndarray, 
                      azimuth: np.ndarray, cv_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation."""
        features = self.extract_polarization_features(dolp, aolp)
        return self.cross_validate_from_features(features, azimuth, cv_folds)
    
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
        instance = cls(**model_data['hyperparameters'])
        instance.regressor = model_data['regressor']
        instance.scaler = model_data['scaler']
        instance.poly_features = model_data['poly_features']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.is_fitted = True
        print(f"Model loaded from {filepath}")
        return instance


def create_baseline_model(alpha: float = 1.0) -> L2PolarizationRegressor:
    """Create the baseline L2 regression model."""
    return L2PolarizationRegressor(
        alpha=alpha,
        polynomial_degree=1,
        include_interactions=False,
        random_state=42
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
