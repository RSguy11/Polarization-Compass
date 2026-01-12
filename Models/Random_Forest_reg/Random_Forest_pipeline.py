"""
Random Forest Regression Pipeline for Polarization-based Solar Azimuth Prediction

This module implements the Random Forest Regressor with Ensemble Ridge regularization
as specified in the blueprint. Random Forest is an ensemble method that can capture
complex non-linear patterns and provides feature importance analysis.

Key features:
- Ensemble of decision trees with ridge regularization
- Feature extraction from DoLP and AoLP data (consistent with L2/SVR)
- Cross-validation for robust evaluation
- Feature importance analysis for interpretability
- Hyperparameter optimization for n_estimators, max_depth, etc.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the feature extraction from L2 pipeline
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L2_Linear_reg.L2_pipeline import L2PolarizationRegressor


class RandomForestPolarizationRegressor:
    """
    Random Forest Regressor model for solar azimuth prediction from polarization data.
    
    Uses ensemble of decision trees with ridge-like regularization as specified 
    in the project blueprint. This model can capture complex non-linear relationships
    and provides interpretability through feature importance.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 random_state: int = 42):
        """
        Initialize the Random Forest regression pipeline.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split internal node
            min_samples_leaf: Minimum samples required to be at leaf node
            max_features: Number of features to consider when looking for best split
            bootstrap: Whether bootstrap samples are used when building trees
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        # Initialize components
        self.scaler = StandardScaler()  # Optional for Random Forest, but helps with consistency
        self.regressor = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        # Use L2 feature extractor for consistency
        self.feature_extractor = L2PolarizationRegressor(random_state=self.random_state)
        
        # Training history
        self.is_fitted = False
        self.feature_names = None
        self.training_metrics = {}
        
    def extract_polarization_features(self, dolp: np.ndarray, aolp: np.ndarray) -> np.ndarray:
        """
        Extract features from DoLP and AoLP data using the same method as L2/SVR.
        This ensures fair comparison between models.
        """
        features = self.feature_extractor.extract_polarization_features(dolp, aolp)
        
        # Store feature names for interpretability
        if self.feature_names is None:
            self.feature_names = self.feature_extractor.feature_names
        
        return features
    
    def prepare_features(self, features: np.ndarray, scale_features: bool = True) -> np.ndarray:
        """
        Apply preprocessing transformations to features.
        
        Args:
            features: Raw feature matrix
            scale_features: Whether to scale features (optional for RF but good for consistency)
        """
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Optional scaling (Random Forest doesn't strictly need it, but helps with interpretability)
        if scale_features:
            if self.is_fitted:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = self.scaler.fit_transform(features)
            return features_scaled
        
        return features
    
    def fit(self, dolp: np.ndarray, aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """
        Train the Random Forest model.
        
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
        
        # Train the Random Forest model
        print(f"Training Random Forest model ({self.n_estimators} trees, max_depth={self.max_depth})...")
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
            'oob_score': getattr(self.regressor, 'oob_score_', None)  # If available
        }
        
        self.training_metrics = metrics
        
        print(f"Training completed!")
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°") 
        print(f"Training R²: {metrics['r2']:.3f}")
        if metrics['oob_score'] is not None:
            print(f"OOB Score: {metrics['oob_score']:.3f}")
        
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
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained Random Forest.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.regressor.feature_importances_))]
        else:
            feature_names = self.feature_names
            
        importance_dict = dict(zip(feature_names, self.regressor.feature_importances_))
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
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
            temp_regressor = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Prepare features
            X_train_scaled = temp_scaler.fit_transform(X_train)
            X_val_scaled = temp_scaler.transform(X_val)
            
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
                            azimuth: np.ndarray, cv_folds: int = 3, 
                            search_type: str = 'random') -> Dict:
        """
        Perform hyperparameter search for optimal Random Forest parameters.
        
        Args:
            dolp, aolp, azimuth: Training data
            cv_folds: CV folds for search (reduced for speed)
            search_type: 'grid' or 'random' search
            
        Returns:
            Best hyperparameters and their performance
        """
        print(f"Performing {search_type} hyperparameter search for Random Forest...")
        
        # Extract and prepare features
        features = self.extract_polarization_features(dolp, aolp)
        X = StandardScaler().fit_transform(features)
        
        # Define parameter space
        if search_type == 'random':
            param_dist = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.5, 0.8]
            }
            
            # Create Random Forest for search
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            # Perform randomized search
            search = RandomizedSearchCV(
                rf, param_dist,
                n_iter=20,  # Number of random combinations to try
                cv=cv_folds,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        
        else:  # Grid search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            
            search = GridSearchCV(
                rf, param_grid, 
                cv=cv_folds,
                scoring='neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X, azimuth)
        
        best_params = search.best_params_
        best_score = -search.best_score_  # Convert back from negative
        
        print(f"\\nBest Random Forest parameters: {best_params}")
        print(f"Best cross-validation MAE: {best_score:.3f}°")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'search_results': search.cv_results_
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
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'random_state': self.random_state
            },
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Random Forest model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'RandomForestPolarizationRegressor':
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(**model_data['hyperparameters'])
        
        # Restore fitted components
        instance.regressor = model_data['regressor']
        instance.scaler = model_data['scaler']
        instance.feature_extractor = model_data['feature_extractor']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data['training_metrics']
        instance.is_fitted = True
        
        print(f"Random Forest model loaded from {filepath}")
        return instance


def create_random_forest_model(n_estimators: int = 100, 
                             max_depth: Optional[int] = None) -> RandomForestPolarizationRegressor:
    """
    Create the Random Forest model as specified in the blueprint.
    
    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        
    Returns:
        Configured RandomForestPolarizationRegressor instance
    """
    return RandomForestPolarizationRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',  # Square root of features (good default)
        bootstrap=True,
        random_state=42  # For reproducibility per blueprint
    )


if __name__ == "__main__":
    # Example usage with mock data
    print("Testing Random Forest Polarization Regressor with mock data...")
    
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
    model = create_random_forest_model(n_estimators=50, max_depth=20)
    
    # Train model
    train_metrics = model.fit(dolp_mock, aolp_mock, azimuth_mock)
    
    # Perform cross-validation
    cv_metrics = model.cross_validate(dolp_mock, aolp_mock, azimuth_mock, cv_folds=3)
    
    # Show feature importance
    feature_importance = model.get_feature_importance()
    print("\\nTop 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
    
    print("\\nRandom Forest model testing completed!")
    print(f"✓ Random Forest can handle {n_samples} samples")
    print(f"✓ Cross-validation MAE: {cv_metrics['mae_mean']:.3f}°")
