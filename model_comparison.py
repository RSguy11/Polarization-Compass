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
    
    def create_test_dataset(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a test dataset for model comparison.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (DoLP, AoLP, azimuth_labels)
        """
        print(f"Creating test dataset with {n_samples} samples...")
        
        # Set seed for reproducibility
        np.random.seed(42)
        
        # Image dimensions
        h, w = 32, 32
        
        # Create realistic DoLP data
        dolp_data = np.random.beta(2, 5, (n_samples, h, w)).astype(np.float32)
        
        # Add spatial structure
        for i in range(n_samples):
            x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            radial = np.sqrt(x**2 + y**2)
            spatial_modulation = 0.2 * np.exp(-radial**2)
            dolp_data[i] += spatial_modulation
            dolp_data[i] = np.clip(dolp_data[i], 0, 1)
        
        # Create AoLP data with gradients
        aolp_data = np.zeros((n_samples, h, w), dtype=np.float32)
        
        for i in range(n_samples):
            base_angle = np.random.uniform(0, 180)
            x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
            gradient = 15 * np.arctan2(y, x) * 180 / np.pi
            
            aolp_data[i] = (base_angle + gradient + np.random.normal(0, 3, (h, w))) % 180
            aolp_data[i] = np.clip(aolp_data[i], 0, 180)
        
        # Create azimuth labels with some correlation
        azimuth_labels = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            # Base from time progression
            base_azimuth = (i / n_samples) * 360
            
            # Add correlation with polarization
            avg_dolp = np.mean(dolp_data[i])
            avg_aolp = np.mean(aolp_data[i])
            
            correlation = 50 * avg_dolp * np.sin(np.deg2rad(2 * avg_aolp))
            noise = np.random.normal(0, 8)  # 8 degree noise
            
            azimuth_labels[i] = (base_azimuth + correlation + noise) % 360
        
        print(f"✓ Test dataset created:")
        print(f"  DoLP: {dolp_data.shape}, range [{dolp_data.min():.3f}, {dolp_data.max():.3f}]")
        print(f"  AoLP: {aolp_data.shape}, range [{aolp_data.min():.1f}°, {aolp_data.max():.1f}°]")
        print(f"  Azimuth: {azimuth_labels.shape}, range [{azimuth_labels.min():.1f}°, {azimuth_labels.max():.1f}°]")
        
        return dolp_data, aolp_data, azimuth_labels
    
    def evaluate_model(self, model, model_name: str, dolp: np.ndarray, 
                      aolp: np.ndarray, azimuth: np.ndarray) -> Dict:
        """
        Evaluate a single model on the test dataset.
        
        Args:
            model: The model instance to evaluate
            model_name: Name of the model for results
            dolp, aolp, azimuth: Test data
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"\\nEvaluating {model_name} model...")
        
        try:
            # Train the model
            print(f"  Training {model_name}...")
            train_metrics = model.fit(dolp, aolp, azimuth)
            
            # Cross-validation
            print(f"  Cross-validating {model_name}...")
            cv_metrics = model.cross_validate(dolp, aolp, azimuth, cv_folds=5)
            
            # Check blueprint compliance
            meets_mae = cv_metrics['mae_mean'] < 5.0
            meets_rmse = cv_metrics['rmse_mean'] <= 10.0
            meets_requirements = meets_mae and meets_rmse
            
            results = {
                'model_name': model_name,
                'training_metrics': train_metrics,
                'cv_metrics': cv_metrics,
                'meets_mae_requirement': meets_mae,
                'meets_rmse_requirement': meets_rmse,
                'meets_blueprint_requirements': meets_requirements,
                'evaluation_success': True
            }
            
            print(f"  ✓ {model_name} evaluation completed")
            print(f"    CV MAE: {cv_metrics['mae_mean']:.3f} ± {cv_metrics['mae_std']:.3f}°")
            print(f"    CV RMSE: {cv_metrics['rmse_mean']:.3f} ± {cv_metrics['rmse_std']:.3f}°")
            print(f"    Meets requirements: {'✓' if meets_requirements else '✗'}")
            
        except Exception as e:
            print(f"  ❌ {model_name} evaluation failed: {str(e)}")
            results = {
                'model_name': model_name,
                'error': str(e),
                'evaluation_success': False
            }
        
        return results
    
    def compare_all_models(self, n_samples: int = 500) -> Dict:
        """
        Compare all three models on identical dataset.
        
        Args:
            n_samples: Number of samples for comparison
            
        Returns:
            Complete comparison results
        """
        print("POLARIZATION COMPASS - MODEL COMPARISON")
        print("=" * 50)
        print(f"Comparing L2, SVR, and Random Forest models")
        print(f"Dataset size: {n_samples} samples")
        
        # Create test dataset
        dolp, aolp, azimuth = self.create_test_dataset(n_samples)
        
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
                    'CV_MAE_Mean': cv['mae_mean'],
                    'CV_MAE_Std': cv['mae_std'],
                    'CV_RMSE_Mean': cv['rmse_mean'],
                    'CV_RMSE_Std': cv['rmse_std'],
                    'CV_R2_Mean': cv['r2_mean'],
                    'Meets_Requirements': results['meets_blueprint_requirements']
                })
                
                print(f"{model_name}:")
                print(f"  MAE: {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}°")
                print(f"  RMSE: {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}°")
                print(f"  R²: {cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}")
                print(f"  Meets requirements: {'✓' if results['meets_blueprint_requirements'] else '✗'}")
                print()
        
        if summary_data:
            # Convert to DataFrame for easy analysis
            df_summary = pd.DataFrame(summary_data)
            
            # Find best performing model
            best_model_idx = df_summary['CV_MAE_Mean'].idxmin()
            best_model = df_summary.iloc[best_model_idx]
            
            # Count models meeting requirements
            models_meeting_req = df_summary['Meets_Requirements'].sum()
            
            self.results['summary'] = {
                'total_models_evaluated': len(summary_data),
                'models_meeting_requirements': int(models_meeting_req),
                'best_model_by_mae': {
                    'name': best_model['Model'],
                    'mae': float(best_model['CV_MAE_Mean']),
                    'rmse': float(best_model['CV_RMSE_Mean']),
                    'r2': float(best_model['CV_R2_Mean'])
                },
                'comparison_table': df_summary.to_dict('records')
            }
            
            print(f"🏆 BEST PERFORMING MODEL: {best_model['Model']}")
            print(f"   MAE: {best_model['CV_MAE_Mean']:.3f}°")
            print(f"   RMSE: {best_model['CV_RMSE_Mean']:.3f}°")
            print(f"   R²: {best_model['CV_R2_Mean']:.3f}")
            
            print(f"\\n📈 BLUEPRINT COMPLIANCE:")
            print(f"   Models meeting requirements: {models_meeting_req}/{len(summary_data)}")
            
            if models_meeting_req > 0:
                print(f"   ✅ SUCCESS: At least one model meets MAE < 5° and RMSE ≤ 10° requirements")
            else:
                print(f"   ⚠️  No models meet blueprint requirements with current test data")
                print(f"   💡 This is expected with mock data. Performance should improve with real polarization data.")
    
    def save_results(self):
        """Save comparison results to files."""
        
        # Save detailed JSON results
        json_path = os.path.join(self.output_dir, 'model_comparison_results.json')
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(self.results, f, indent=2, default=convert_types)
        
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
        
        with open(report_path, 'w') as f:
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
                
                if 'best_model_by_mae' in summary:
                    best = summary['best_model_by_mae']
                    f.write(f"BEST MODEL (lowest MAE): {best['name']}\\n")
                    f.write(f"  MAE: {best['mae']:.3f}°\\n")
                    f.write(f"  RMSE: {best['rmse']:.3f}°\\n")
                    f.write(f"  R²: {best['r2']:.3f}\\n\\n")
                
                f.write("BLUEPRINT REQUIREMENTS (MAE < 5°, RMSE ≤ 10°):\\n")
                f.write("-" * 45 + "\\n")
                
                for model_name, results in self.results['models'].items():
                    if results.get('evaluation_success', False):
                        meets = results['meets_blueprint_requirements']
                        cv = results['cv_metrics']
                        f.write(f"{model_name}: {'✓' if meets else '✗'} ")
                        f.write(f"(MAE: {cv['mae_mean']:.3f}°, RMSE: {cv['rmse_mean']:.3f}°)\\n")
            
            f.write("\\n" + "=" * 60 + "\\n")
            f.write("END OF REPORT")
        
        print(f"📄 Detailed report saved to {report_path}")


def main():
    """Run the complete model comparison."""
    
    # Create comparison suite
    comparison = ModelComparison(output_dir="model_comparison_results")
    
    # Run comparison with manageable dataset size
    results = comparison.compare_all_models(n_samples=300)
    
    print(f"\\n🎯 MODEL COMPARISON COMPLETE!")
    print(f"📂 Check 'model_comparison_results/' for detailed results")
    print(f"\\n🚀 NEXT STEPS:")
    print("1. Connect real polarization data from your preprocessing pipeline")
    print("2. Add actual solar azimuth labels using the solar_azimuth_generator.py")
    print("3. Re-run comparison with real data to see which model performs best")
    print("4. Use the best performing model for your underwater polarization compass")


if __name__ == "__main__":
    main()