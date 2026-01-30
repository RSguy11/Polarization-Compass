import os
import sys
import numpy as np
from datetime import datetime
import json
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def extract_statistical_features_from_single_image(dolp, aolp):
    """
    Extract ~500 statistical + spatial features from DoLP and AoLP data.
    Uses 9x9 grid (81 regions) for maximum spatial resolution.
    
    Features breakdown:
    - DoLP global stats: mean, std, median, max, min, skew, kurtosis (7)
    - AoLP global stats: mean, std, median, max, min, skew, kurtosis (7)
    - Circular AoLP: cos(2*AoLP) mean/std, sin(2*AoLP) mean/std (4)
    - Cross products: DoLP*cos(2*AoLP) mean, DoLP*sin(2*AoLP) mean (2)
    - 9x9 Grid DoLP means: 81 regions (81)
    - 9x9 Grid AoLP cos(2*aolp) means: 81 regions (81)
    - 9x9 Grid AoLP sin(2*aolp) means: 81 regions (81)
    - 9x9 Grid DoLP*cos cross: 81 regions (81)
    - 9x9 Grid DoLP*sin cross: 81 regions (81)
    - Gradient features: h/v gradients for DoLP and AoLP (4)
    - Center vs edge: DoLP center, edge, ratio; AoLP center, edge, ratio (6)
    - Percentiles: DoLP 10/25/50/75/90, AoLP 10/25/50/75/90 (10)
    - Weighted centroids: DoLP-weighted x,y centroid (2)
    - Radial features: DoLP by distance from center (3)
    - Row/column profiles: 9 rows, 9 cols (18)
    
    Total: ~468 features
    """
    h, w = dolp.shape
    dolp = dolp.astype(np.float64)
    aolp = aolp.astype(np.float64)
    
    # Flatten for global statistics
    dolp_flat = dolp.ravel()
    aolp_flat = aolp.ravel()
    
    # ===== GLOBAL STATISTICS (14 features) =====
    # Basic statistics for DoLP (7 features including skew/kurtosis)
    dolp_mean = np.mean(dolp_flat)
    dolp_std = np.std(dolp_flat) + 1e-8
    dolp_features = [
        dolp_mean,
        dolp_std,
        np.median(dolp_flat),
        np.max(dolp_flat),
        np.min(dolp_flat),
        np.mean(((dolp_flat - dolp_mean) / dolp_std) ** 3),  # skewness
        np.mean(((dolp_flat - dolp_mean) / dolp_std) ** 4) - 3  # excess kurtosis
    ]
    
    # Basic statistics for AoLP (7 features)
    aolp_mean = np.mean(aolp_flat)
    aolp_std = np.std(aolp_flat) + 1e-8
    aolp_features = [
        aolp_mean,
        aolp_std,
        np.median(aolp_flat),
        np.max(aolp_flat),
        np.min(aolp_flat),
        np.mean(((aolp_flat - aolp_mean) / aolp_std) ** 3),  # skewness
        np.mean(((aolp_flat - aolp_mean) / aolp_std) ** 4) - 3  # excess kurtosis
    ]
    
    # Circular/trigonometric features for AoLP (4 features)
    aolp_rad = np.deg2rad(aolp_flat)
    cos2_aolp = np.cos(2 * aolp_rad)
    sin2_aolp = np.sin(2 * aolp_rad)
    
    circular_features = [
        np.mean(cos2_aolp),
        np.mean(sin2_aolp),
        np.std(cos2_aolp),
        np.std(sin2_aolp)
    ]
    
    # Cross-product features (2 features)
    cross_features = [
        np.mean(dolp_flat * cos2_aolp),
        np.mean(dolp_flat * sin2_aolp)
    ]
    
    # ===== 3x3 GRID SPATIAL FEATURES (45 features) =====
    # Split image into 3x3 grid for finer spatial resolution
    aolp_rad_2d = np.deg2rad(aolp)
    cos2_aolp_2d = np.cos(2 * aolp_rad_2d)
    sin2_aolp_2d = np.sin(2 * aolp_rad_2d)
    
    grid_dolp = []
    grid_aolp_cos = []
    grid_aolp_sin = []
    grid_cross_cos = []
    grid_cross_sin = []
    
    grid_size = 9  # 9x9 grid = 81 regions
    h_step, w_step = h // grid_size, w // grid_size
    for i in range(grid_size):  # rows
        for j in range(grid_size):  # cols
            h_start, h_end = i * h_step, (i + 1) * h_step if i < grid_size - 1 else h
            w_start, w_end = j * w_step, (j + 1) * w_step if j < grid_size - 1 else w
            
            region_dolp = dolp[h_start:h_end, w_start:w_end]
            region_cos = cos2_aolp_2d[h_start:h_end, w_start:w_end]
            region_sin = sin2_aolp_2d[h_start:h_end, w_start:w_end]
            
            grid_dolp.append(np.mean(region_dolp))
            grid_aolp_cos.append(np.mean(region_cos))
            grid_aolp_sin.append(np.mean(region_sin))
            grid_cross_cos.append(np.mean(region_dolp * region_cos))
            grid_cross_sin.append(np.mean(region_dolp * region_sin))
    
    # ===== GRADIENT FEATURES (4 features) =====
    # Captures directional trends across the image
    dolp_grad_h = np.mean(dolp[:, -1]) - np.mean(dolp[:, 0])  # left-to-right trend
    dolp_grad_v = np.mean(dolp[-1, :]) - np.mean(dolp[0, :])  # top-to-bottom trend
    aolp_grad_h = np.mean(cos2_aolp_2d[:, -1]) - np.mean(cos2_aolp_2d[:, 0])
    aolp_grad_v = np.mean(cos2_aolp_2d[-1, :]) - np.mean(cos2_aolp_2d[0, :])
    gradient_features = [dolp_grad_h, dolp_grad_v, aolp_grad_h, aolp_grad_v]
    
    # ===== CENTER VS EDGE FEATURES (6 features) =====
    # Define center region (middle 50%) and edge region
    h_quarter, w_quarter = h // 4, w // 4
    center_mask = np.zeros_like(dolp, dtype=bool)
    center_mask[h_quarter:3*h_quarter, w_quarter:3*w_quarter] = True
    edge_mask = ~center_mask
    
    dolp_center = np.mean(dolp[center_mask])
    dolp_edge = np.mean(dolp[edge_mask])
    dolp_center_edge_ratio = dolp_center / (dolp_edge + 1e-8)
    
    aolp_center_cos = np.mean(cos2_aolp_2d[center_mask])
    aolp_edge_cos = np.mean(cos2_aolp_2d[edge_mask])
    aolp_center_edge_diff = aolp_center_cos - aolp_edge_cos
    
    center_edge_features = [dolp_center, dolp_edge, dolp_center_edge_ratio, 
                           aolp_center_cos, aolp_edge_cos, aolp_center_edge_diff]
    
    # ===== PERCENTILE FEATURES (10 features) =====
    percentile_features = [
        np.percentile(dolp_flat, 10),
        np.percentile(dolp_flat, 25),
        np.percentile(dolp_flat, 50),
        np.percentile(dolp_flat, 75),
        np.percentile(dolp_flat, 90),
        np.percentile(aolp_flat, 10),
        np.percentile(aolp_flat, 25),
        np.percentile(aolp_flat, 50),
        np.percentile(aolp_flat, 75),
        np.percentile(aolp_flat, 90)
    ]
    
    # ===== WEIGHTED CENTROID (2 features) =====
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    dolp_sum = np.sum(dolp) + 1e-8
    centroid_x = np.sum(x_coords * dolp) / dolp_sum / w  # Normalized 0-1
    centroid_y = np.sum(y_coords * dolp) / dolp_sum / h  # Normalized 0-1
    centroid_features = [centroid_x, centroid_y]
    
    # ===== RADIAL FEATURES (3 features) =====
    # DoLP by distance from center
    center_y, center_x = h // 2, w // 2
    y_dist = (y_coords - center_y).astype(np.float64) / h
    x_dist = (x_coords - center_x).astype(np.float64) / w
    radius = np.sqrt(y_dist**2 + x_dist**2)
    
    inner_mask = radius < 0.25
    mid_mask = (radius >= 0.25) & (radius < 0.4)
    outer_mask = radius >= 0.4
    
    radial_features = [
        np.mean(dolp[inner_mask]) if np.any(inner_mask) else 0,
        np.mean(dolp[mid_mask]) if np.any(mid_mask) else 0,
        np.mean(dolp[outer_mask]) if np.any(outer_mask) else 0
    ]
    
    # ===== ROW/COLUMN PROFILE FEATURES (18 features) =====
    # Average DoLP in 9 horizontal strips and 9 vertical strips
    row_ninth = h // 9
    col_ninth = w // 9
    profile_features = []
    for i in range(9):
        h_start = i * row_ninth
        h_end = (i + 1) * row_ninth if i < 8 else h
        profile_features.append(np.mean(dolp[h_start:h_end, :]))
    for i in range(9):
        w_start = i * col_ninth
        w_end = (i + 1) * col_ninth if i < 8 else w
        profile_features.append(np.mean(dolp[:, w_start:w_end]))
    
    # ===== COMBINE ALL FEATURES (~468 total) =====
    all_features = (
        dolp_features +           # 7
        aolp_features +           # 7
        circular_features +       # 4
        cross_features +          # 2
        grid_dolp +               # 81
        grid_aolp_cos +           # 81
        grid_aolp_sin +           # 81
        grid_cross_cos +          # 81
        grid_cross_sin +          # 81
        gradient_features +       # 4
        center_edge_features +    # 6
        percentile_features +     # 10
        centroid_features +       # 2
        radial_features +         # 3
        profile_features          # 18
    )  # Total: 468
    
    features = np.array(all_features, dtype=np.float32)
    
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class CircularRegressionWrapper:
    """
    Wrapper that handles circular nature of azimuth (0° = 360°).
    
    Instead of predicting azimuth directly, predicts sin(azimuth) and cos(azimuth)
    separately, then converts back. This eliminates the discontinuity at 0°/360°.
    
    Supports optional PCA dimensionality reduction and feature selection.
    """
    
    def __init__(self, base_model_class, use_pca=False, n_components=100, 
                 use_feature_selection=False, k_best=200, **model_kwargs):
        from sklearn.preprocessing import StandardScaler
        
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.scaler = StandardScaler()
        
        # Optional PCA
        self.use_pca = use_pca
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) if use_pca else None
        
        # Optional feature selection
        self.use_feature_selection = use_feature_selection
        self.k_best = k_best
        self.feature_selector = None  # Will be fit during training
        
        # Create two separate models: one for sin, one for cos
        self.sin_model = base_model_class(**model_kwargs)
        self.cos_model = base_model_class(**model_kwargs)
        
        self.is_fitted = False
        self.training_metrics = {}
    
    def fit_from_features(self, features, azimuth, sample_weight=None):
        """Train on sin/cos targets instead of raw azimuth."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        print(f"Training CIRCULAR regression with {features.shape[0]} samples, {features.shape[1]} features")
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        
        # Optional feature selection (before PCA)
        if self.use_feature_selection:
            self.feature_selector = SelectKBest(mutual_info_regression, k=min(self.k_best, X.shape[1]))
            # Use sin(azimuth) as proxy for feature selection
            X = self.feature_selector.fit_transform(X, np.sin(azimuth))
            print(f"  Feature selection: {features.shape[1]} → {X.shape[1]} features")
        
        # Optional PCA dimensionality reduction
        if self.use_pca:
            n_comp = min(self.n_components, X.shape[1], X.shape[0])
            self.pca = PCA(n_components=n_comp)
            X = self.pca.fit_transform(X)
            print(f"  PCA: reduced to {X.shape[1]} components (explained var: {self.pca.explained_variance_ratio_.sum():.1%})")
        
        # Convert azimuth to sin/cos targets
        sin_target = np.sin(azimuth)
        cos_target = np.cos(azimuth)
        
        print(f"Training sin model...")
        self.sin_model.fit(X, sin_target, sample_weight=sample_weight)
        print(f"Training cos model...")
        self.cos_model.fit(X, cos_target, sample_weight=sample_weight)
        
        self.is_fitted = True
        
        # Evaluate
        y_pred = self.predict_from_features(features)
        
        # Calculate circular MAE (handles wraparound)
        diff = np.abs(azimuth - y_pred)
        diff = np.minimum(diff, 2*np.pi - diff)  # Handle circular wraparound
        
        metrics = {
            'mae': np.rad2deg(np.mean(diff)),
            'rmse': np.rad2deg(np.sqrt(np.mean(diff**2))),
            'n_samples': len(azimuth),
            'n_features': X.shape[1]
        }
        self.training_metrics = metrics
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°")
        return metrics
    
    def predict_from_features(self, features):
        """Predict azimuth by combining sin/cos predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        
        # Apply same transformations as training
        if self.use_feature_selection and self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        if self.use_pca and self.pca is not None:
            X = self.pca.transform(X)
        
        sin_pred = self.sin_model.predict(X)
        cos_pred = self.cos_model.predict(X)
        
        # Convert back to angle using atan2
        azimuth_pred = np.arctan2(sin_pred, cos_pred)
        # Ensure positive angles (0 to 2π)
        azimuth_pred = np.where(azimuth_pred < 0, azimuth_pred + 2*np.pi, azimuth_pred)
        
        return azimuth_pred
    
    def cross_validate_from_features(self, features, azimuth, cv_folds=5):
        """Perform k-fold cross-validation."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        mae_scores = []
        
        print(f"Performing {cv_folds}-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
            temp_model = CircularRegressionWrapper(
                self.base_model_class, 
                use_pca=self.use_pca, n_components=self.n_components,
                use_feature_selection=self.use_feature_selection, k_best=self.k_best,
                **self.model_kwargs
            )
            temp_model.fit_from_features(features[train_idx], azimuth[train_idx])
            y_pred = temp_model.predict_from_features(features[val_idx])
            
            diff = np.abs(azimuth[val_idx] - y_pred)
            diff = np.minimum(diff, 2*np.pi - diff)
            mae_scores.append(np.rad2deg(np.mean(diff)))
            print(f"  Fold {fold}/{cv_folds}: MAE: {mae_scores[-1]:.3f}°")
        
        results = {'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores)}
        print(f"\nCV MAE: {results['mae_mean']:.3f} ± {results['mae_std']:.3f}°")
        return results
    
    def save(self, filepath):
        import joblib
        joblib.dump({'scaler': self.scaler, 'sin_model': self.sin_model, 'cos_model': self.cos_model}, filepath)


class EnsembleCircularModel:
    """
    Ensemble of multiple circular regression models.
    Combines predictions using weighted averaging based on validation performance.
    """
    
    def __init__(self, models_config):
        """
        models_config: list of (name, model_class, kwargs) tuples
        """
        self.models_config = models_config
        self.models = {}
        self.weights = {}
        self.is_fitted = False
        self.training_metrics = {}
    
    def fit_from_features(self, features, azimuth, sample_weight=None):
        """Train all models and compute ensemble weights based on training error."""
        from sklearn.model_selection import cross_val_score, KFold
        
        print(f"Training ENSEMBLE with {len(self.models_config)} models...")
        
        # First pass: train all models and get their CV errors
        cv_errors = {}
        for name, model_class, kwargs in self.models_config:
            print(f"\n  Training {name}...")
            model = CircularRegressionWrapper(model_class, **kwargs)
            model.fit_from_features(features, azimuth, sample_weight)
            self.models[name] = model
            
            # Quick CV to estimate error for weighting
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            errors = []
            for train_idx, val_idx in kf.split(features):
                temp = CircularRegressionWrapper(model_class, **kwargs)
                temp.fit_from_features(features[train_idx], azimuth[train_idx])
                pred = temp.predict_from_features(features[val_idx])
                diff = np.abs(azimuth[val_idx] - pred)
                diff = np.minimum(diff, 2*np.pi - diff)
                errors.append(np.mean(diff))
            cv_errors[name] = np.mean(errors)
            print(f"    CV error: {np.rad2deg(cv_errors[name]):.2f}°")
        
        # Compute weights: inverse of error (better models get higher weight)
        total_inv_error = sum(1.0 / (e + 0.01) for e in cv_errors.values())
        for name, error in cv_errors.items():
            self.weights[name] = (1.0 / (error + 0.01)) / total_inv_error
            print(f"  {name} weight: {self.weights[name]:.3f}")
        
        self.is_fitted = True
        
        # Evaluate ensemble
        y_pred = self.predict_from_features(features)
        diff = np.abs(azimuth - y_pred)
        diff = np.minimum(diff, 2*np.pi - diff)
        
        metrics = {
            'mae': np.rad2deg(np.mean(diff)),
            'rmse': np.rad2deg(np.sqrt(np.mean(diff**2))),
            'n_samples': len(azimuth),
            'n_features': features.shape[1]
        }
        self.training_metrics = metrics
        print(f"\nEnsemble Training MAE: {metrics['mae']:.3f}°")
        return metrics
    
    def predict_from_features(self, features):
        """Weighted average of all model predictions (in sin/cos space)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Average in sin/cos space for better circular handling
        sin_sum = np.zeros(features.shape[0])
        cos_sum = np.zeros(features.shape[0])
        
        for name, model in self.models.items():
            pred = model.predict_from_features(features)
            weight = self.weights[name]
            sin_sum += weight * np.sin(pred)
            cos_sum += weight * np.cos(pred)
        
        # Convert back to angle
        azimuth_pred = np.arctan2(sin_sum, cos_sum)
        azimuth_pred = np.where(azimuth_pred < 0, azimuth_pred + 2*np.pi, azimuth_pred)
        return azimuth_pred
    
    def cross_validate_from_features(self, features, azimuth, cv_folds=5):
        """Perform k-fold cross-validation on ensemble."""
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        mae_scores = []
        
        print(f"Performing {cv_folds}-fold CV on ensemble...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
            temp_ensemble = EnsembleCircularModel(self.models_config)
            temp_ensemble.fit_from_features(features[train_idx], azimuth[train_idx])
            y_pred = temp_ensemble.predict_from_features(features[val_idx])
            
            diff = np.abs(azimuth[val_idx] - y_pred)
            diff = np.minimum(diff, 2*np.pi - diff)
            mae_scores.append(np.rad2deg(np.mean(diff)))
            print(f"  Fold {fold}/{cv_folds}: MAE: {mae_scores[-1]:.3f}°")
        
        results = {'mae_mean': np.mean(mae_scores), 'mae_std': np.std(mae_scores)}
        print(f"\nEnsemble CV MAE: {results['mae_mean']:.3f} ± {results['mae_std']:.3f}°")
        return results
    
    def save(self, filepath):
        import joblib
        joblib.dump({'models': self.models, 'weights': self.weights}, filepath)


class GradientBoostingWrapper:
    """Wrapper for sklearn GradientBoostingRegressor with same interface as other models."""
    
    def __init__(self, n_estimators=200, max_depth=5, learning_rate=0.1, 
                 subsample=0.8, min_samples_split=5, min_samples_leaf=2):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        
        self.scaler = StandardScaler()
        self.regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_iter_no_change=20,  # Early stopping
            validation_fraction=0.1
        )
        self.is_fitted = False
        self.training_metrics = {}
    
    def fit_from_features(self, features, azimuth, sample_weight=None):
        """Train the Gradient Boosting model."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        print(f"Training with {features.shape[0]} samples, {features.shape[1]} features")
        print("Preprocessing features...")
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.fit_transform(X)
        print(f"Final feature matrix shape: {X.shape}")
        print(f"Training Gradient Boosting (n_estimators={self.n_estimators}, max_depth={self.max_depth}, lr={self.learning_rate})...")
        
        self.regressor.fit(X, azimuth, sample_weight=sample_weight)
        self.is_fitted = True
        
        y_pred = self.regressor.predict(X)
        metrics = {
            'mae': np.rad2deg(mean_absolute_error(azimuth, y_pred)),
            'rmse': np.rad2deg(np.sqrt(mean_squared_error(azimuth, y_pred))),
            'r2': self.regressor.score(X, azimuth),
            'n_samples': len(azimuth),
            'n_features': X.shape[1]
        }
        self.training_metrics = metrics
        print(f"Training completed!")
        print(f"Training MAE: {metrics['mae']:.3f}°")
        print(f"Training RMSE: {metrics['rmse']:.3f}°")
        print(f"Training R²: {metrics['r2']:.3f}")
        print(f"Trees used: {self.regressor.n_estimators_}")
        return metrics
    
    def predict_from_features(self, features):
        """Predict azimuth from features."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(X)
        return self.regressor.predict(X)
    
    def cross_validate_from_features(self, features, azimuth, cv_folds=5):
        """Perform k-fold cross-validation."""
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        mae_scores, rmse_scores, r2_scores = [], [], []
        
        print(f"Performing {cv_folds}-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(features), 1):
            temp_model = GradientBoostingWrapper(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample
            )
            temp_model.fit_from_features(features[train_idx], azimuth[train_idx])
            y_pred = temp_model.predict_from_features(features[val_idx])
            
            mae_scores.append(np.rad2deg(mean_absolute_error(azimuth[val_idx], y_pred)))
            rmse_scores.append(np.rad2deg(np.sqrt(mean_squared_error(azimuth[val_idx], y_pred))))
            r2_scores.append(temp_model.regressor.score(
                temp_model.scaler.transform(np.nan_to_num(features[val_idx], nan=0.0, posinf=0.0, neginf=0.0)),
                azimuth[val_idx]
            ))
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
    
    def save(self, filepath):
        """Save model to file."""
        import joblib
        joblib.dump({'scaler': self.scaler, 'regressor': self.regressor}, filepath)
        print(f"Gradient Boosting model saved to {filepath}")


def load_batch_features(loader, indices, verbose=False):
    """Load batch and extract 468 STATISTICAL+SPATIAL features from DoLP and AoLP."""
    feature_list = []
    failed_count = 0
    
    for idx_num, idx in enumerate(indices):
        if verbose and (idx_num + 1) % 50 == 0:
            print(f"    Processing {idx_num + 1}/{len(indices)} features (extracted: {len(feature_list)}, failed: {failed_count})...")
            sys.stdout.flush()
        
        try:
            sample = loader.get_item(idx)
            if sample is not None:
                features = extract_statistical_features_from_single_image(
                    sample['features']['dolp'],
                    sample['features']['aolp']
                )
                # Features are already float32, 16 values
                feature_list.append(features)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
        
        # Force garbage collection every 50 samples to free memory from temporary arrays
        if (idx_num + 1) % 50 == 0:
            gc.collect()
    
    print(f"    Successfully extracted {len(feature_list)} features ({failed_count} failed)")
    
    # Convert to numpy array with float32 to save memory
    if len(feature_list) > 0:
        return np.array(feature_list, dtype=np.float32)
    else:
        raise ValueError("No features were successfully extracted!")


def calculate_sample_weights(azimuth_labels, n_bins=8):
    """
    Calculate sample weights based on azimuth distribution.
    Rare azimuths get higher weights to balance training.
    
    Args:
        azimuth_labels: Array of azimuth angles (0-360°)
        n_bins: Number of bins to divide 360° into
        
    Returns:
        weights: Array of normalized weights
    """
    # Bin azimuths into n_bins bins
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_indices = np.digitize(azimuth_labels, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Count samples per bin
    unique_bins, bin_counts = np.unique(bin_indices, return_counts=True)
    
    # Calculate weights: inverse frequency (rare bins get high weights)
    # Avoid division by zero by using safe division
    max_count = bin_counts.max()
    weights = np.zeros_like(azimuth_labels, dtype=float)
    
    for bin_idx, count in zip(unique_bins, bin_counts):
        mask = bin_indices == bin_idx
        # Weight = max_count / count (higher for rarer bins)
        weight_value = max_count / max(count, 1)  # Prevent division by zero
        weights[mask] = weight_value
    
    # Normalize to [0.5, 2.0] range to avoid extreme weights
    if weights.max() > weights.min():
        weights = 0.5 + 1.5 * (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = np.ones_like(weights)  # All equal if no variation
    
    # Ensure no NaN or inf values
    weights = np.nan_to_num(weights, nan=1.0, posinf=2.0, neginf=0.5)
    
    return weights


def run_complete_pipeline():
    """Run the complete model training pipeline for all models with circular regression."""
    
    # IMPORT SKLEARN AND MODELS HERE (after __main__ guard)
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from solar_azimuth_generator import SolarPositionCalculator
    from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader
    from Training_loops.visualization import create_training_plots
    
    print("POLARIZATION COMPASS - COMPLETE MODEL PIPELINE")
    print("=" * 60)
    print("Training L2, SVR, Random Forest, and Gradient Boosting models")
    print()
    
    print("STEP 1: Loading & Caching Data")
    print("=" * 60)
    
    rmc_folder = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/Polarization_DataLoader/rmc")
    loader = PolarizationDataLoader(rmc_folder=rmc_folder)
    max_samples = len(loader)
    
    batch_size = 100
    all_indices = []
    all_labels = []
    
    print(f"Loading metadata from {max_samples} samples...")
    for i in range(max_samples):
        try:
            all_indices.append(i)
            all_labels.append(loader.labels_df.iloc[i]['azimuth'])
            
            # Progress every 300
            if (i + 1) % 300 == 0:
                print(f"  Loaded {i + 1}/{max_samples} samples...")
        except Exception as e:
            continue
    
    print(f"[OK] Loaded metadata for {len(all_indices)} samples")
    print(f"    Azimuth range: {min(all_labels):.1f}° → {max(all_labels):.1f}°")
    
    azimuth = np.deg2rad(np.array(all_labels))
    n_samples = len(all_indices)
    
    

    # STRATIFIED AZIMUTH SPLIT: ensure test set covers all azimuths
    print("\nStratified azimuth splitting (test set covers all azimuths)...")
    azimuth_deg = np.rad2deg(azimuth)
    n_bins = 18  # 20-degree bins
    bins = np.linspace(0, 360, n_bins+1)
    az_bin = np.digitize(azimuth_deg, bins, right=False) - 1
    # Clamp last bin
    az_bin[az_bin == n_bins] = n_bins-1
    test_indices = []
    train_indices = []
    rng = np.random.default_rng(42)
    for b in range(n_bins):
        bin_idxs = [i for i, ab in enumerate(az_bin) if ab == b]
        if len(bin_idxs) == 0:
            continue
        n_test = max(1, int(0.2 * len(bin_idxs)))
        test_in_bin = rng.choice(bin_idxs, size=n_test, replace=False)
        train_in_bin = [i for i in bin_idxs if i not in test_in_bin]
        test_indices.extend([all_indices[i] for i in test_in_bin])
        train_indices.extend([all_indices[i] for i in train_in_bin])
    # Shuffle for randomness
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    azimuth_train_deg = np.array([all_labels[i] for i in train_indices])
    azimuth_test_deg = np.array([all_labels[i] for i in test_indices])
    # Convert to radians for model training
    azimuth_train = np.deg2rad(azimuth_train_deg)
    azimuth_test = np.deg2rad(azimuth_test_deg)
    train_size = len(train_indices)
    test_size = len(test_indices)
    print(f"[OK] Train: {train_size} samples")
    print(f"[OK] Test: {test_size} samples")
    print(f"    Train azimuth range: {min(azimuth_train_deg):.1f}° → {max(azimuth_train_deg):.1f}°")
    print(f"    Test azimuth range: {min(azimuth_test_deg):.1f}° → {max(azimuth_test_deg):.1f}°")
    
    print("\nCaching training features (468 SPATIAL+STATISTICAL features per sample)...")
    all_train_features = load_batch_features(loader, train_indices, verbose=True)
    print(f"[OK] Cached training features: {all_train_features.shape}")
    
    print("Caching test features...")
    all_test_features = load_batch_features(loader, test_indices, verbose=True)
    print(f"[OK] Cached test features: {all_test_features.shape}\n")
    
    today = datetime.now().strftime('%Y-%m-%d')
    results = {}
    training_history = {}
    
    print("STEP 2: Training Models")
    print("=" * 60)
    
    # Import sklearn models for circular regression
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    
    models = {
        # L2 with PCA to reduce dimensionality (helps linear models)
        'L2_PCA': CircularRegressionWrapper(
            Ridge,
            use_pca=True,
            n_components=50,  # Reduce 468 features to 50 principal components
            alpha=0.1
        ),
        
        # SVR Circular - the original that achieved 2.46° (RESTORED)
        'SVR_Circular': CircularRegressionWrapper(
            SVR,
            C=50.0,
            gamma='scale',   # Auto-scale gamma
            epsilon=0.01,
            kernel='rbf'
        ),
        
        # RF with more trees and deeper for best performance
        'RF_Enhanced': CircularRegressionWrapper(
            RandomForestRegressor,
            n_estimators=500,   # More trees
            max_depth=20,       # Deeper
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        
        # Ensemble of best models
        'Ensemble': EnsembleCircularModel([
            ('SVR', SVR, {'C': 50.0, 'gamma': 'scale', 'epsilon': 0.01, 'kernel': 'rbf'}),
            ('RF', RandomForestRegressor, {'n_estimators': 300, 'max_depth': 15, 'random_state': 42, 'n_jobs': -1}),
            ('Ridge', Ridge, {'alpha': 0.1}),
        ]),
        
        # Keep Gradient Boosting as reference
        'Gradient_Boosting': GradientBoostingWrapper(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.9,
            min_samples_split=3,
            min_samples_leaf=1
        )
    }
    
    # Calculate sample weights based on azimuth distribution
    print("\nCalculating sample weights for balanced training...")
    # Note: calculate_sample_weights expects DEGREES (0-360), so convert from radians
    train_weights = calculate_sample_weights(np.rad2deg(azimuth_train))
    print(f"[OK] Sample weights calculated")
    print(f"    Weight range: {train_weights.min():.3f} - {train_weights.max():.3f}")
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        try:
            # Skip learning curves for Ensemble (too slow)
            if 'Ensemble' in model_name:
                print(f"  Skipping learning curve for ensemble model...")
                best_val_error = 0.0
                best_val_rmse = 0.0
                best_sample_size = train_size
                train_errors = []
                val_errors = []
            else:
                print(f"  Generating learning curve...")
                
                if train_size >= 200:
                    sample_sizes = [50, 100, 200, train_size]  # Reduced for speed
                else:
                    sample_sizes = [min(50, train_size), train_size]
                
                sample_sizes = sorted(list(set(sample_sizes)))
                train_errors = []
                val_errors = []
                best_val_error = float('inf')
                best_val_rmse = float('inf')
                best_sample_size = 0
                
                for size in sample_sizes:
                    # Create temporary model based on model type
                    if 'L2' in model_name or 'PCA' in model_name:
                        model_temp = CircularRegressionWrapper(Ridge, use_pca='PCA' in model_name, n_components=50, alpha=0.1)
                    elif 'SVR' in model_name:
                        model_temp = CircularRegressionWrapper(
                            SVR, C=100.0, gamma=0.01, epsilon=0.001, kernel='rbf'
                        )
                    elif 'Gradient' in model_name:
                        model_temp = GradientBoostingWrapper(
                            n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9
                        )
                    else:  # RF
                        model_temp = CircularRegressionWrapper(
                            RandomForestRegressor, n_estimators=200, max_depth=15,
                            min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1
                        )
                
                subset_train_size = int(size * 0.8)
                subset_val_size = size - subset_train_size
                subset_indices = np.random.choice(range(len(all_train_features)), subset_train_size, replace=False)
                train_features = all_train_features[subset_indices]
                subset_weights = train_weights[subset_indices]
                
                train_metrics_temp = model_temp.fit_from_features(
                    train_features, 
                    azimuth_train[subset_indices],
                    sample_weight=subset_weights
                )
                if subset_val_size > 0:
                    remaining_indices = np.setdiff1d(np.arange(len(all_train_features)), subset_indices)
                    val_indices = np.random.choice(remaining_indices, subset_val_size, replace=False)
                    val_features = all_train_features[val_indices]
                    
                    val_pred = model_temp.predict_from_features(val_features)
                    val_error = np.rad2deg(np.mean(np.abs(val_pred - azimuth_train[val_indices])))
                    val_rmse = np.rad2deg(np.sqrt(np.mean((val_pred - azimuth_train[val_indices])**2)))
                else:
                    val_error = train_metrics_temp['mae']
                    val_rmse = train_metrics_temp['rmse']
                
                train_errors.append(train_metrics_temp['mae'])
                val_errors.append(val_error)
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_val_rmse = val_rmse
                    best_sample_size = size
            
            print(f"  Best validation MAE: {best_val_error:.3f} deg at {best_sample_size} samples")
            
            training_history[model_name] = {
                'sample_sizes': sample_sizes,
                'train_errors': train_errors,
                'val_errors': val_errors,
                'best_val_error': best_val_error,
                'best_val_rmse': best_val_rmse,
                'best_sample_size': best_sample_size
            }
            
            print(f"  Training final model on full training set ({train_size} samples)...")
            train_metrics = model.fit_from_features(all_train_features, azimuth_train, sample_weight=train_weights)
            
            print(f"  Evaluating on test set ({test_size} samples)...")
            test_predictions = model.predict_from_features(all_test_features)
            test_mae = np.rad2deg(np.mean(np.abs(test_predictions - azimuth_test)))
            test_rmse = np.rad2deg(np.sqrt(np.mean((test_predictions - azimuth_test) ** 2)))
            
            cv_metrics = model.cross_validate_from_features(all_train_features, azimuth_train, cv_folds=5)
            
            # Handle both circular and non-circular model metrics
            cv_rmse = cv_metrics.get('rmse_mean', 0.0)
            cv_std = cv_metrics.get('mae_std', 0.0)
            
            results[model_name] = {
                'training_mae': float(train_metrics['mae']),
                'cv_mae': float(cv_metrics['mae_mean']),
                'cv_mae_std': float(cv_std),
                'cv_rmse': float(cv_rmse),
                'best_val_mae': float(best_val_error),
                'best_val_rmse': float(best_val_rmse),
                'best_val_samples': int(best_sample_size),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'meets_requirements': bool(test_mae < 5.0)
            }
            
            print(f" {model_name}")
            print(f"  Train MAE: {train_metrics['mae']:.3f} deg")
            print(f"  CV MAE: {cv_metrics['mae_mean']:.3f} deg")
            print(f"  Test MAE: {test_mae:.3f} deg (held-out)")
            
            model_dir = os.path.join('saved_models', today)
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{model_name}_final.pkl')
            if hasattr(model, 'save_model'):
                model.save_model(model_path)
            elif hasattr(model, 'save'):
                model.save(model_path)
            else:
                import joblib
                joblib.dump(model, model_path)
                print(f"{model_name} model saved to {model_path}")
            print(f"  Final model saved: {model_path}")
            
        except Exception as e:
            print(f" {model_name} failed: {str(e)}")
            results[model_name] = {'error': str(e)}
            import traceback
            traceback.print_exc()
    
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = os.path.join('training_plots', today)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'complete_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n RESULTS SUMMARY:")
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name}:")
            print(f"  CV MAE: {result['cv_mae']:.3f} deg")
            print(f"  Test MAE: {result['test_mae']:.3f} deg - {'PASS' if result['meets_requirements'] else 'FAIL'}")
    
    print(f"\nGenerating training plots...")
    create_training_plots(results, training_history, results_dir)
    print(f"\nResults and plots saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    run_complete_pipeline()