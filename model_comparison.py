"""
Model Comparison and Evaluation Suite

This script compares all three regression models as specified in the blueprint:
1. L2 Linear Regression (baseline)
2. SVR with RBF kernel  
3. Random Forest with ensemble regularization

It evaluates their performance on identical datasets and determines which
meets the blueprint requirements (MAE < 5°, RMSE ≤ 10%).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import all model pipelines
sys.path.append('.')
from L2_Linear_reg.L2_pipeline import create_baseline_model
from SVR_reg.SVR_pipeline import create_svr_model
from Random_Forest_reg.Random_Forest_pipeline import create_random_forest_model

# Import Ben's spatial PNG data loader  
from Bens_Data_Import.Image_data_loader.SpatialPolarizationLoader import SpatialPolarizationLoader
from pathlib import Path


def angular_difference(pred_deg, true_deg):
    """
    Compute the minimum angular difference between predicted and true angles.
    
    Args:
        pred_deg: Predicted angles in degrees
        true_deg: True angles in degrees
        
    Returns:
        Angular differences in degrees, range [-180, 180]
    """
    diff = pred_deg - true_deg
    # Wrap to [-180, 180] range
    diff = (diff + 180) % 360 - 180
    return diff


def angular_mae(pred_deg, true_deg):
    """Compute Mean Absolute Error for angular data."""
    angular_diffs = angular_difference(pred_deg, true_deg)
    return np.mean(np.abs(angular_diffs))


def angular_rmse(pred_deg, true_deg):
    """Compute Root Mean Square Error for angular data."""
    angular_diffs = angular_difference(pred_deg, true_deg)
    return np.sqrt(np.mean(angular_diffs**2))


def circular_correlation(pred_deg, true_deg):
    """
    Compute circular correlation coefficient for angular data.
    
    Returns:
        Circular correlation coefficient [-1, 1]
    """
    # Convert to radians
    pred_rad = np.deg2rad(pred_deg)
    true_rad = np.deg2rad(true_deg)
    
    # Compute complex representations
    pred_complex = np.exp(1j * pred_rad)
    true_complex = np.exp(1j * true_rad)
    
    # Circular correlation
    numerator = np.abs(np.mean(pred_complex * np.conj(true_complex)))**2
    denominator = np.mean(np.abs(pred_complex)**2) * np.mean(np.abs(true_complex)**2)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def compute_angular_metrics(pred_deg, true_deg):
    """
    Compute comprehensive angular error metrics.
    
    Args:
        pred_deg: Predicted azimuth angles in degrees
        true_deg: True azimuth angles in degrees
        
    Returns:
        Dictionary of angular metrics
    """
    return {
        'angular_mae': angular_mae(pred_deg, true_deg),
        'angular_rmse': angular_rmse(pred_deg, true_deg),
        'circular_correlation': circular_correlation(pred_deg, true_deg),
        'angular_diffs': angular_difference(pred_deg, true_deg),
        'max_angular_error': np.max(np.abs(angular_difference(pred_deg, true_deg))),
        'angular_std': np.std(angular_difference(pred_deg, true_deg))
    }


class ModelComparison:
    """
    Comprehensive model comparison and evaluation suite.
    
    Compares L2, SVR, and Random Forest models on identical datasets
    following the blueprint requirements.
    """
    
    def __init__(self, output_dir: str = "model_comparison_results"):
        """
        Initialize the model comparison suite.
        
        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {}
        }
        
        print(f"Model Comparison Suite initialized")
        print(f"Results will be saved to: {output_dir}")
    
    def load_real_polarization_data(self, use_spatial: bool = True, target_size: tuple = (64, 64)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load Ben's real PNG polarization dataset with spatial information preserved.
        
        Args:
            use_spatial: Whether to use spatial features or statistical features
            target_size: Target spatial resolution for processing
        
        Returns:
            Tuple of (DoLP_data, AoLP_data, azimuth_labels) with spatial patterns preserved
        """
        print("Loading SPATIAL polarization data from Ben's PNG dataset...")
        
        # Path to Ben's PNG dataset
        png_data_path = Path("C:/Queens/ELEC498/Ben's Data/24-10-08-t000-forward-paradesquare/24-10-08-t000-forward-paradesquare")
        
        # Initialize spatial PNG database loader
        loader = SpatialPolarizationLoader(
            data_path=png_data_path, 
            start_deg=0.0, 
            step_deg=1.0,
            target_size=target_size
        )
        
        # Load spatial data
        dolp_spatial, aolp_spatial, azimuth_labels = loader.get_spatial_data()
        
        # Convert azimuth labels from radians to degrees
        azimuth_degrees = np.rad2deg(azimuth_labels) % 360
        
        print(f"✓ Spatial polarization dataset loaded:")
        print(f"  Samples: {len(dolp_spatial)}")
        print(f"  Spatial resolution: {target_size}")
        print(f"  DoLP spatial range: [{dolp_spatial.min():.3f}, {dolp_spatial.max():.3f}]")
        print(f"  AoLP spatial range: [{aolp_spatial.min():.1f}°, {aolp_spatial.max():.1f}°]")
        print(f"  Azimuth range: [{azimuth_degrees.min():.1f}°, {azimuth_degrees.max():.1f}°]")
        
        # Check spatial variation
        dolp_vars = [np.var(dolp_spatial[i]) for i in range(min(5, len(dolp_spatial)))]
        aolp_vars = [np.var(aolp_spatial[i]) for i in range(min(5, len(aolp_spatial)))]
        print(f"  Average spatial variation - DoLP: {np.mean(dolp_vars):.6f}, AoLP: {np.mean(aolp_vars):.1f}")
        
        return dolp_spatial, aolp_spatial, azimuth_degrees
    
    def evaluate_model(self, model, model_name: str, dolp: np.ndarray, 
                      aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """
        Evaluate a single model on the test dataset with angular metrics.
        
        Args:
            model: The model instance to evaluate
            model_name: Name of the model for results
            dolp, aolp, azimuth: Test data
            
        Returns:
            Dictionary containing evaluation results with angular metrics
        """
        print(f"\\nEvaluating {model_name} model...")
        
        try:
            # Train the model
            print(f"  Training {model_name}...")
            train_metrics = model.fit(dolp, aolp, azimuth)
            
            # Cross-validation with angular metrics
            print(f"  Cross-validating {model_name} with angular metrics...")
            cv_results = self._cross_validate_angular(model, dolp, aolp, azimuth, cv_folds=5)
            
            # Check blueprint compliance using angular metrics
            meets_angular_mae = cv_results['angular_mae_mean'] < 5.0
            meets_angular_rmse = cv_results['angular_rmse_mean'] <= 10.0
            meets_requirements = meets_angular_mae and meets_angular_rmse
            
            results = {
                'model_name': model_name,
                'training_metrics': train_metrics,
                'cv_metrics': cv_results,
                'meets_angular_mae_requirement': meets_angular_mae,
                'meets_angular_rmse_requirement': meets_angular_rmse,
                'meets_blueprint_requirements': meets_requirements,
                'evaluation_success': True
            }
            
            print(f"  ✓ {model_name} evaluation completed")
            print(f"    Angular MAE: {cv_results['angular_mae_mean']:.3f} ± {cv_results['angular_mae_std']:.3f}°")
            print(f"    Angular RMSE: {cv_results['angular_rmse_mean']:.3f} ± {cv_results['angular_rmse_std']:.3f}°")
            print(f"    Circular Correlation: {cv_results['circular_corr_mean']:.3f} ± {cv_results['circular_corr_std']:.3f}")
            print(f"    Meets requirements: {'✓' if meets_requirements else '✗'}")
            
        except Exception as e:
            print(f"  ❌ {model_name} evaluation failed: {str(e)}")
            results = {
                'model_name': model_name,
                'error': str(e),
                'evaluation_success': False
            }
        
        return results
    
    def _cross_validate_angular(self, model, dolp: np.ndarray, aolp: np.ndarray, 
                               azimuth: np.ndarray, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation with angular error metrics.
        
        Returns:
            Dictionary of angular cross-validation metrics
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        angular_maes = []
        angular_rmses = []
        circular_corrs = []
        linear_maes = []  # For comparison
        linear_rmses = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dolp)):
            # Split data
            dolp_train, dolp_val = dolp[train_idx], dolp[val_idx]
            aolp_train, aolp_val = aolp[train_idx], aolp[val_idx]
            azimuth_train, azimuth_val = azimuth[train_idx], azimuth[val_idx]
            
            # Train model on fold
            temp_model = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            temp_model.fit(dolp_train, aolp_train, azimuth_train)
            
            # Predict on validation set
            pred_azimuth = temp_model.predict(dolp_val, aolp_val)
            
            # Compute angular metrics
            angular_metrics = compute_angular_metrics(pred_azimuth, azimuth_val)
            angular_maes.append(angular_metrics['angular_mae'])
            angular_rmses.append(angular_metrics['angular_rmse'])
            circular_corrs.append(angular_metrics['circular_correlation'])
            
            # Compute linear metrics for comparison
            linear_maes.append(np.mean(np.abs(pred_azimuth - azimuth_val)))
            linear_rmses.append(np.sqrt(np.mean((pred_azimuth - azimuth_val)**2)))
        
        return {
            'angular_mae_mean': np.mean(angular_maes),
            'angular_mae_std': np.std(angular_maes),
            'angular_rmse_mean': np.mean(angular_rmses),
            'angular_rmse_std': np.std(angular_rmses),
            'circular_corr_mean': np.mean(circular_corrs),
            'circular_corr_std': np.std(circular_corrs),
            'linear_mae_mean': np.mean(linear_maes),  # For comparison
            'linear_rmse_mean': np.mean(linear_rmses),
            'angular_maes': angular_maes,
            'angular_rmses': angular_rmses,
            'circular_corrs': circular_corrs
        }
    
    def compare_all_models(self) -> Dict:
        """
        Compare all three models using Ben's real PNG polarization dataset.
        
        Returns:
            Complete comparison results
        """
        print("POLARIZATION COMPASS - MODEL COMPARISON")
        print("=" * 50)
        print(f"Comparing L2, SVR, and Random Forest models")
        print("Using SPATIAL polarization data from Ben's PNG dataset")
        
        # Load spatial dataset
        dolp, aolp, azimuth = self.load_real_polarization_data(target_size=(64, 64))
        
        print(f"Dataset size: {len(dolp)} samples")
        
        # Initialize models
        models = {
            'L2_Linear_Regression': create_baseline_model(alpha=1.0),
            'SVR_RBF': create_svr_model(C=1.0, gamma='scale', epsilon=0.1),
            'Random_Forest': create_random_forest_model(n_estimators=50, max_depth=20)  # Reduced for speed
        }
        
        # Evaluate each model
        for model_name, model in models.items():
            results = self.evaluate_model(model, model_name, dolp, aolp, azimuth)
            self.results['models'][model_name] = results
        
        # Create comparison summary
        self.create_comparison_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def create_comparison_summary(self):
        """Create a summary comparison of all models."""
        
        print("\\n📊 MODEL COMPARISON SUMMARY")
        print("=" * 30)
        
        summary_data = []
        
        for model_name, results in self.results['models'].items():
            if results.get('evaluation_success', False):
                cv = results['cv_metrics']
                summary_data.append({
                    'Model': model_name,
                    'Angular_MAE_Mean': cv['angular_mae_mean'],
                    'Angular_MAE_Std': cv['angular_mae_std'],
                    'Angular_RMSE_Mean': cv['angular_rmse_mean'],
                    'Angular_RMSE_Std': cv['angular_rmse_std'],
                    'Circular_Corr_Mean': cv['circular_corr_mean'],
                    'Linear_MAE_Mean': cv['linear_mae_mean'],  # For comparison
                    'Linear_RMSE_Mean': cv['linear_rmse_mean'],
                    'Meets_Requirements': results['meets_blueprint_requirements']
                })
                
                print(f"{model_name}:")
                print(f"  Angular MAE: {cv['angular_mae_mean']:.3f} ± {cv['angular_mae_std']:.3f}°")
                print(f"  Angular RMSE: {cv['angular_rmse_mean']:.3f} ± {cv['angular_rmse_std']:.3f}°")
                print(f"  Circular Correlation: {cv['circular_corr_mean']:.3f} ± {cv['circular_corr_std']:.3f}")
                print(f"  Linear MAE (comparison): {cv['linear_mae_mean']:.3f}°")
                print(f"  Meets requirements: {'✓' if results['meets_blueprint_requirements'] else '✗'}")
                print()
        
        if summary_data:
            # Convert to DataFrame for easy analysis
            df_summary = pd.DataFrame(summary_data)
            
            # Find best performing model (by angular MAE)
            best_model_idx = df_summary['Angular_MAE_Mean'].idxmin()
            best_model = df_summary.iloc[best_model_idx]
            
            # Count models meeting requirements
            models_meeting_req = df_summary['Meets_Requirements'].sum()
            
            self.results['summary'] = {
                'total_models_evaluated': len(summary_data),
                'models_meeting_requirements': int(models_meeting_req),
                'best_model_by_angular_mae': {
                    'name': best_model['Model'],
                    'angular_mae': float(best_model['Angular_MAE_Mean']),
                    'angular_rmse': float(best_model['Angular_RMSE_Mean']),
                    'circular_corr': float(best_model['Circular_Corr_Mean']),
                    'linear_mae_comparison': float(best_model['Linear_MAE_Mean'])
                },
                'comparison_table': df_summary.to_dict('records')
            }
            
            print(f"🏆 BEST PERFORMING MODEL (Angular MAE): {best_model['Model']}")
            print(f"   Angular MAE: {best_model['Angular_MAE_Mean']:.3f}°")
            print(f"   Angular RMSE: {best_model['Angular_RMSE_Mean']:.3f}°")
            print(f"   Circular Correlation: {best_model['Circular_Corr_Mean']:.3f}")
            print(f"   Linear MAE (for comparison): {best_model['Linear_MAE_Mean']:.3f}°")
            
            print(f"\\n📈 BLUEPRINT COMPLIANCE:")
            print(f"   Models meeting requirements: {models_meeting_req}/{len(summary_data)}")
            
            if models_meeting_req > 0:
                print(f"   ✅ SUCCESS: At least one model meets angular MAE < 5° and RMSE ≤ 10° requirements")
            else:
                print(f"   ⚠️  No models meet blueprint requirements")
                print(f"   💡 Angular metrics show true compass performance vs misleading linear errors")
    
    def save_results(self):
        """Save comparison results to files."""
        
        # Save detailed JSON results
        json_path = os.path.join(self.output_dir, 'model_comparison_results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                return obj
            
            # Create a copy of results without circular references
            results_copy = {}
            for key, value in self.results.items():
                if key == 'models':
                    models_copy = {}
                    for model_name, model_results in value.items():
                        if isinstance(model_results, dict) and model_results.get('evaluation_success', False):
                            # Only keep serializable data
                            models_copy[model_name] = {
                                'model_name': model_results['model_name'],
                                'cv_metrics': model_results.get('cv_metrics', {}),
                                'meets_mae_requirement': bool(model_results.get('meets_mae_requirement', False)),
                                'meets_rmse_requirement': bool(model_results.get('meets_rmse_requirement', False)),
                                'meets_blueprint_requirements': bool(model_results.get('meets_blueprint_requirements', False)),
                                'evaluation_success': bool(model_results.get('evaluation_success', False))
                            }
                        else:
                            models_copy[model_name] = model_results
                    results_copy[key] = models_copy
                else:
                    results_copy[key] = value
            
            json.dump(results_copy, f, indent=2, default=convert_types)
        
        print(f"\\n💾 Results saved to {json_path}")
        
        # Save summary CSV
        if 'comparison_table' in self.results['summary']:
            csv_path = os.path.join(self.output_dir, 'model_comparison_summary.csv')
            df = pd.DataFrame(self.results['summary']['comparison_table'])
            df.to_csv(csv_path, index=False)
            print(f"💾 Summary table saved to {csv_path}")
        
        # Create human-readable report
        self.create_text_report()
    
    def create_text_report(self):
        """Create a human-readable text report."""
        
        report_path = os.path.join(self.output_dir, 'comparison_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("POLARIZATION COMPASS - MODEL COMPARISON REPORT\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write(f"Comparison completed: {self.results['comparison_timestamp']}\\n\\n")
            
            f.write("MODELS EVALUATED:\\n")
            f.write("-" * 20 + "\\n")
            for model_name in self.results['models'].keys():
                f.write(f"✓ {model_name}\\n")
            f.write("\\n")
            
            if 'summary' in self.results:
                summary = self.results['summary']
                
                f.write("PERFORMANCE SUMMARY:\\n")
                f.write("-" * 20 + "\\n")
                f.write(f"Total models evaluated: {summary['total_models_evaluated']}\\n")
                f.write(f"Models meeting blueprint requirements: {summary['models_meeting_requirements']}\\n\\n")
                
                if 'best_model_by_angular_mae' in summary:
                    best = summary['best_model_by_angular_mae']
                    f.write(f"BEST MODEL (lowest Angular MAE): {best['name']}\\n")
                    f.write(f"  Angular MAE: {best['angular_mae']:.3f}°\\n")
                    f.write(f"  Angular RMSE: {best['angular_rmse']:.3f}°\\n")
                    f.write(f"  Circular Correlation: {best['circular_corr']:.3f}\\n")
                    f.write(f"  Linear MAE (comparison): {best['linear_mae_comparison']:.3f}°\\n\\n")
                
                f.write("BLUEPRINT REQUIREMENTS (Angular MAE < 5°, RMSE ≤ 10°):\\n")
                f.write("-" * 50 + "\\n")
                
                for model_name, results in self.results['models'].items():
                    if results.get('evaluation_success', False):
                        meets = results['meets_blueprint_requirements']
                        cv = results['cv_metrics']
                        f.write(f"{model_name}: {'✓' if meets else '✗'} ")
                        f.write(f"(Angular MAE: {cv['angular_mae_mean']:.3f}°, Angular RMSE: {cv['angular_rmse_mean']:.3f}°)\\n")
            
            f.write("\\n" + "=" * 60 + "\\n")
            f.write("END OF REPORT")
        
        print(f"📄 Detailed report saved to {report_path}")


def main():
    """Run the complete model comparison."""
    
    # Create comparison suite
    comparison = ModelComparison(output_dir="model_comparison_results")
    
    print("🔬 RUNNING MODEL COMPARISON WITH SPATIAL POLARIZATION DATA")
    results = comparison.compare_all_models()
    
    print(f"\\n🎯 MODEL COMPARISON COMPLETE!")
    print(f"📂 Check 'model_comparison_results/' for detailed results")
    print(f"\\n🚀 NEXT STEPS:")
    print("1. ✅ Using SPATIAL polarization data (64×64 patterns per image)")
    print("2. Compare performance improvement vs previous scalar approach")
    print("3. Analyze which model best captures spatial polarization patterns")
    print("4. Consider increasing spatial resolution if needed (128×128)")
    
    return results


if __name__ == "__main__":
    main()