"""SVR (Support Vector Regression) Pipeline for Polarization-based Solar Azimuth Prediction"""

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L2_Linear_reg.L2_pipeline import L2PolarizationRegressor


class SVRPolarizationRegressor:
    """SVR model for solar azimuth prediction from polarization data."""
    
    def __init__(self, 
                 C: float = 1.0,
                 gamma: str = 'scale',
                 epsilon: float = 0.1,
                 kernel: str = 'rbf',
                 random_state: int = 42):
        """Initialize the SVR regression pipeline."""
        self.C = C
        self.gamma = gamma
        self.epsilon = epsilon
        self.kernel = kernel
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.regressor = SVR(C=self.C, gamma=self.gamma, epsilon=self.epsilon, kernel=self.kernel)
        self.feature_extractor = L2PolarizationRegressor(random_state=self.random_state)
        self.is_fitted = False
        self.training_metrics = {}
        
    def extract_polarization_features(self, dolp: np.ndarray, aolp: np.ndarray) -> np.ndarray:
        """Extract features using L2 method."""
        return self.feature_extractor.extract_polarization_features(dolp, aolp)
    
    def prepare_features(self, features: np.ndarray) -> np.ndarray:
        """Apply preprocessing transformations."""
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.is_fitted:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = self.scaler.fit_transform(features)
        return features_scaled
    
    def fit_from_features(self, features: np.ndarray, azimuth: np.ndarray) -> Dict:
        """Train the SVR model using pre-extracted features."""
        print(f"Training with {features.shape[0]} samples, {features.shape[1]} features")
        print("Preprocessing features...")
        X = self.prepare_features(features)
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Training SVR model (kernel={self.kernel}, C={self.C}, gamma={self.gamma})...")
        
        self.regressor.fit(X, azimuth)
        self.is_fitted = True
        
        y_pred = self.regressor.predict(X)
        
        metrics = {
            'mae': np.rad2deg(mean_absolute_error(azimuth, y_pred)),
            'rmse': np.rad2deg(np.sqrt(mean_squared_error(azimuth, y_pred))),
            'r2': self.regressor.score(X, azimuth),
            'n_samples': len(azimuth),
            'n_features': X.shape[1],
            'n_support_vectors': len(self.regressor.support_)
        }
        
        self.training_metrics = metrics
        print(f"Training completed!")
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°") 
        print(f"Training R²: {metrics['r2']:.3f}")
        print(f"Support vectors: {metrics['n_support_vectors']}")
        
        return metrics
    
    def predict_from_features(self, features: np.ndarray) -> np.ndarray:
        """Predict solar azimuth from pre-extracted features."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self.prepare_features(features)
        return self.regressor.predict(X)
    
    def cross_validate_from_features(self, features: np.ndarray, azimuth: np.ndarray, cv_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation using pre-extracted features."""
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        mae_scores, rmse_scores, r2_scores = [], [], []
        
        print(f"Performing {cv_folds}-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
            temp_model = SVRPolarizationRegressor(C=self.C, gamma=self.gamma, epsilon=self.epsilon, kernel=self.kernel)
            temp_model.fit_from_features(features[train_idx], azimuth[train_idx])
            y_pred = temp_model.predict_from_features(features[val_idx])
            
            mae_scores.append(np.rad2deg(mean_absolute_error(azimuth[val_idx], y_pred)))
            rmse_scores.append(np.rad2deg(np.sqrt(mean_squared_error(azimuth[val_idx], y_pred))))
            r2_scores.append(temp_model.regressor.score(temp_model.prepare_features(features[val_idx]), azimuth[val_idx]))
            
            print(f"  Fold {fold}/{cv_folds}")
            print(f"    MAE: {mae_scores[-1]:.3f}°, RMSE: {rmse_scores[-1]:.3f}°, R²: {r2_scores[-1]:.3f}")
        
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
    
    def fit(self, dolp: np.ndarray, aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """
        Train the SVR model.
        
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
        
        # Train the SVR model
        print(f"Training SVR model (kernel={self.kernel}, C={self.C}, gamma={self.gamma})...")
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
            'n_features': X.shape[1],
            'n_support_vectors': len(self.regressor.support_)
        }
        
        self.training_metrics = metrics
        
        print(f"Training completed!")
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°") 
        print(f"Training R²: {metrics['r2']:.3f}")
        print(f"Support vectors: {metrics['n_support_vectors']}")
        
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
            temp_regressor = SVR(
                C=self.C,
                gamma=self.gamma,
                epsilon=self.epsilon,
                kernel=self.kernel
            )
            
            # Prepare features
            X_train_scaled = temp_scaler.fit_transform(X_train)
            X_val_scaled = temp_scaler.transform(X_val)
            
            # Train and predict
            temp_regressor.fit(X_train_scaled, y_train)
            y_pred = temp_regressor.predict(X_val_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
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
        
        print(f"\\nCross-validation Results:")
        print(f"MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}°")
        print(f"RMSE: {cv_results['rmse_mean']:.3f} ± {cv_results['rmse_std']:.3f}°")
        print(f"R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        
        # Check if meets blueprint requirements (MAE < 5°, RMSE ≤ 10%)
        meets_requirements = (cv_results['mae_mean'] < 5.0 and 
                            cv_results['rmse_mean'] <= 10.0)
        
        print(f"Meets blueprint requirements: {'✓' if meets_requirements else '✗'}")
        
        return cv_results
    
    def hyperparameter_search(self, dolp: np.ndarray, aolp: np.ndarray, 
                            azimuth: np.ndarray, cv_folds: int = 3) -> Dict:
        """
        Perform grid search for optimal hyperparameters.
        
        Args:
            dolp, aolp, azimuth: Training data
            cv_folds: CV folds for grid search (reduced for speed)
            
        Returns:
            Best hyperparameters and their performance
        """
        print("Performing hyperparameter search for SVR...")
        
        # Extract and prepare features
        features = self.extract_polarization_features(dolp, aolp)
        X = StandardScaler().fit_transform(features)
        
        # Define parameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'epsilon': [0.01, 0.1, 0.2, 0.5]
        }
        
        # Create SVR for grid search
        svr = SVR(kernel=self.kernel)
        
        # Perform grid search
        grid_search = GridSearchCV(
            svr, param_grid, 
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, azimuth)
        
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_  # Convert back from negative
        
        print(f"\\nBest SVR parameters: {best_params}")
        print(f"Best cross-validation MAE: {best_score:.3f}°")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'grid_search_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        model_data = {
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'hyperparameters': {
                'C': self.C,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'kernel': self.kernel,
                'random_state': self.random_state
            },
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"SVR model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'SVRPolarizationRegressor':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(**model_data['hyperparameters'])
        
        # Restore fitted components
        instance.regressor = model_data['regressor']
        instance.scaler = model_data['scaler']
        instance.feature_extractor = model_data['feature_extractor']
        instance.training_metrics = model_data['training_metrics']
        instance.is_fitted = True
        
        print(f"SVR model loaded from {filepath}")
        return instance


def create_svr_model(C: float = 1.0, gamma: str = 'scale', epsilon: float = 0.1) -> SVRPolarizationRegressor:
    """
    Create the SVR model as specified in the blueprint.
    
    Args:
        C: Regularization parameter
        gamma: Kernel coefficient
        epsilon: Epsilon-tube parameter
        
    Returns:
        Configured SVRPolarizationRegressor instance
    """
    return SVRPolarizationRegressor(
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        kernel='rbf',  # RBF kernel as specified in blueprint
        random_state=42  # For reproducibility per blueprint
    )


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing SVR Polarization Regressor with mock data...")
    
    # Create mock data (replace with real data loading)
    n_samples = 200  # Smaller dataset for testing
    h, w = 32, 32
    
    # Mock DoLP and AoLP data
    np.random.seed(42)
    dolp_mock = np.random.uniform(0, 1, (n_samples, h, w))
    aolp_mock = np.random.uniform(0, 180, (n_samples, h, w))
    
    # Mock azimuth labels (target variable)
    azimuth_mock = np.random.uniform(0, 360, n_samples)
    
    # Create and test model
    model = create_svr_model(C=1.0, gamma='scale', epsilon=0.1)
    
    # Train model
    train_metrics = model.fit(dolp_mock, aolp_mock, azimuth_mock)
    
    # Perform cross-validation
    cv_metrics = model.cross_validate(dolp_mock, aolp_mock, azimuth_mock, cv_folds=3)
    
    print("\\nSVR model testing completed!")
    print(f"✓ SVR can handle {n_samples} samples")
    print(f"✓ Cross-validation MAE: {cv_metrics['mae_mean']:.3f}°")
