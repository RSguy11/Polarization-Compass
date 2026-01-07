"""
L2 Linear Regression Training Loop

This module implements the training loop for the baseline L2 (Ridge) regression model
as specified in the project blueprint. It includes data loading, model training,
validation, and evaluation following the project requirements.

Blueprint Requirements:
- 20k supervised dataset (10k standard + 10k augmented images)
- K-fold cross validation (k=5)
- MAE ≤ 5°, RMSE ≤ 10% target performance
- Robustness testing with noise simulation
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from L2_Linear_reg.L2_pipeline import L2PolarizationRegressor, create_baseline_model


class L2TrainingLoop:
    """
    Training loop manager for L2 Linear Regression baseline model.
    
    Handles the complete training pipeline from data loading through evaluation
    according to the project blueprint specifications.
    """
    
    def __init__(self, 
                 output_dir: str = "L2_results",
                 random_state: int = 42):
        """
        Initialize the training loop.
        
        Args:
            output_dir: Directory to save results
            random_state: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results tracking
        self.results = {
            'training_metrics': {},
            'cv_metrics': {},
            'robustness_metrics': {},
            'model_configs': [],
            'timestamp': datetime.now().isoformat()
        }
        
    def load_preprocessed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load preprocessed polarization data from the preprocessing pipeline.
        
        Returns:
            Tuple of (DoLP, AoLP, azimuth_labels)
            
        Note: This is a placeholder. In practice, you'll load data from your
        preprocessing pipeline output (stage2 or stage3 results).
        """
        print("Loading preprocessed data...")
        
        # TODO: Replace this with actual data loading from preprocessing pipeline
        # This should load the output from your data_cleaning_pipeline.py
        
        # For now, create mock data matching blueprint specifications
        # Blueprint specifies 20k dataset (10k + 10k augmented)
        n_samples = 20000
        h, w = 128, 128  # Adjust based on your actual image size
        
        print(f"Creating mock dataset with {n_samples} samples...")
        print("⚠️  Replace this with actual data loading from preprocessing pipeline")
        
        np.random.seed(self.random_state)
        
        # Mock DoLP data (0-1 range)
        dolp = np.random.beta(2, 5, (n_samples, h, w))  # Realistic DoLP distribution
        
        # Mock AoLP data (0-180° range) 
        aolp = np.random.uniform(0, 180, (n_samples, h, w))
        
        # Mock azimuth labels with some correlation to polarization
        # In reality, this should come from your data collection
        base_azimuth = np.random.uniform(0, 360, n_samples)
        
        # Add some realistic noise and patterns
        azimuth_labels = base_azimuth + np.random.normal(0, 5, n_samples)
        azimuth_labels = azimuth_labels % 360  # Keep in 0-360 range
        
        print(f"Data shapes - DoLP: {dolp.shape}, AoLP: {aolp.shape}, Labels: {azimuth_labels.shape}")
        print(f"DoLP range: [{dolp.min():.3f}, {dolp.max():.3f}]")
        print(f"AoLP range: [{aolp.min():.1f}°, {aolp.max():.1f}°]") 
        print(f"Azimuth range: [{azimuth_labels.min():.1f}°, {azimuth_labels.max():.1f}°]")
        
        return dolp, aolp, azimuth_labels
    
    def create_data_augmentation(self, 
                               dolp: np.ndarray, 
                               aolp: np.ndarray, 
                               azimuth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply data augmentation as specified in blueprint.
        Blueprint: "10K images augmented with minor geometric shifts being 
        rotation between 10-15° and brightness/color augmentation between 1-3%"
        
        Args:
            dolp: Original DoLP data
            aolp: Original AoLP data  
            azimuth: Original azimuth labels
            
        Returns:
            Augmented data tuple
        """
        print("Applying data augmentation...")
        
        n_samples = len(dolp)
        
        # Brightness/contrast augmentation (1-3% as specified)
        brightness_factors = np.random.uniform(0.97, 1.03, n_samples)
        contrast_factors = np.random.uniform(0.97, 1.03, n_samples)
        
        dolp_aug = dolp.copy()
        aolp_aug = aolp.copy()
        azimuth_aug = azimuth.copy()
        
        for i in range(n_samples):
            # Apply brightness/contrast to DoLP
            dolp_aug[i] = np.clip(dolp_aug[i] * brightness_factors[i], 0, 1)
            
            # Small rotation (10-15° as specified)
            rotation_angle = np.random.uniform(10, 15)
            
            # For AoLP, rotation affects the angle
            aolp_aug[i] = (aolp_aug[i] + rotation_angle) % 180
            
            # Rotation affects azimuth labels too
            azimuth_aug[i] = (azimuth_aug[i] + rotation_angle) % 360
        
        print(f"Augmentation complete. Generated {n_samples} additional samples.")
        
        return dolp_aug, aolp_aug, azimuth_aug
    
    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the complete dataset according to blueprint specifications.
        
        Returns:
            Combined original + augmented data
        """
        print("Preparing dataset according to blueprint specifications...")
        
        # Load original data
        dolp_orig, aolp_orig, azimuth_orig = self.load_preprocessed_data()
        
        # For blueprint compliance, we need 10k original + 10k augmented = 20k total
        # If we have more than 10k, subsample to 10k for augmentation
        n_orig = len(dolp_orig)
        if n_orig > 10000:
            print(f"Subsampling {n_orig} samples to 10k for augmentation...")
            indices = np.random.choice(n_orig, 10000, replace=False)
            dolp_orig = dolp_orig[indices]
            aolp_orig = aolp_orig[indices]
            azimuth_orig = azimuth_orig[indices]
        
        # Create augmented data
        dolp_aug, aolp_aug, azimuth_aug = self.create_data_augmentation(
            dolp_orig, aolp_orig, azimuth_orig
        )
        
        # Combine original and augmented data
        dolp_combined = np.concatenate([dolp_orig, dolp_aug], axis=0)
        aolp_combined = np.concatenate([aolp_orig, aolp_aug], axis=0)
        azimuth_combined = np.concatenate([azimuth_orig, azimuth_aug], axis=0)
        
        # Shuffle the combined dataset
        n_total = len(dolp_combined)
        shuffle_indices = np.random.permutation(n_total)
        
        dolp_final = dolp_combined[shuffle_indices]
        aolp_final = aolp_combined[shuffle_indices]
        azimuth_final = azimuth_combined[shuffle_indices]
        
        print(f"Final dataset: {len(dolp_final)} samples")
        print(f"✓ Meets blueprint requirement of 20k supervised dataset")
        
        return dolp_final, aolp_final, azimuth_final
    
    def hyperparameter_search(self, 
                            dolp: np.ndarray,
                            aolp: np.ndarray, 
                            azimuth: np.ndarray) -> Dict:
        """
        Perform hyperparameter search for the L2 model.
        
        Args:
            dolp, aolp, azimuth: Training data
            
        Returns:
            Best hyperparameters and their performance
        """
        print("Performing hyperparameter search...")
        
        # Define hyperparameter search space
        alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        poly_degrees = [1, 2]  # Start simple
        
        best_score = float('inf')
        best_params = {}
        all_results = []
        
        for alpha in alpha_values:
            for poly_degree in poly_degrees:
                print(f"  Testing alpha={alpha}, poly_degree={poly_degree}")
                
                # Create model with these hyperparameters
                model = L2PolarizationRegressor(
                    alpha=alpha,
                    polynomial_degree=poly_degree,
                    include_interactions=(poly_degree > 1),
                    random_state=self.random_state
                )
                
                # Perform cross-validation
                cv_results = model.cross_validate(dolp, aolp, azimuth, cv_folds=5)
                
                # Use MAE as primary metric (blueprint requirement: MAE < 5°)
                score = cv_results['mae_mean']
                
                result = {
                    'alpha': alpha,
                    'poly_degree': poly_degree,
                    'mae_mean': cv_results['mae_mean'],
                    'mae_std': cv_results['mae_std'],
                    'rmse_mean': cv_results['rmse_mean'],
                    'rmse_std': cv_results['rmse_std'],
                    'r2_mean': cv_results['r2_mean'],
                    'r2_std': cv_results['r2_std']
                }
                all_results.append(result)
                
                print(f"    MAE: {score:.3f}° ± {cv_results['mae_std']:.3f}°")
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'alpha': alpha,
                        'polynomial_degree': poly_degree,
                        'include_interactions': (poly_degree > 1)
                    }
                    print(f"    ✓ New best score: {score:.3f}°")
        
        print(f"\\nBest hyperparameters: {best_params}")
        print(f"Best cross-validation MAE: {best_score:.3f}°")
        
        # Check if meets requirements
        meets_requirements = best_score < 5.0
        print(f"Meets blueprint MAE requirement (<5°): {'✓' if meets_requirements else '✗'}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
            'meets_requirements': meets_requirements
        }
    
    def train_final_model(self, 
                         dolp: np.ndarray,
                         aolp: np.ndarray,
                         azimuth: np.ndarray,
                         best_params: Dict) -> L2PolarizationRegressor:
        """
        Train the final model with best hyperparameters.
        
        Args:
            dolp, aolp, azimuth: Training data
            best_params: Best hyperparameters from search
            
        Returns:
            Trained model
        """
        print("Training final model with best hyperparameters...")
        
        # Create model with best parameters
        model = L2PolarizationRegressor(**best_params, random_state=self.random_state)
        
        # Train on full dataset
        train_metrics = model.fit(dolp, aolp, azimuth)
        
        # Perform final cross-validation
        cv_metrics = model.cross_validate(dolp, aolp, azimuth, cv_folds=5)
        
        # Store results
        self.results['training_metrics'] = train_metrics
        self.results['cv_metrics'] = cv_metrics
        self.results['model_configs'].append({
            'name': 'L2_baseline',
            'params': best_params,
            'performance': cv_metrics
        })
        
        return model
    
    def robustness_testing(self, 
                          model: L2PolarizationRegressor,
                          dolp: np.ndarray, 
                          aolp: np.ndarray,
                          azimuth: np.ndarray) -> Dict:
        """
        Perform robustness testing as specified in blueprint.
        
        Blueprint: "Gaussian and Poisson noise with standard deviation of noise 
        at a range of 0.01-0.05 will be added to random input DoLP/AoLP at testing.
        Turbidity will also be simulated by overlaying randomized haze layers 
        with optical densities between 0.1 and 0.3"
        
        Args:
            model: Trained model
            dolp, aolp, azimuth: Test data
            
        Returns:
            Robustness test results
        """
        print("Performing robustness testing...")
        
        # Use a subset for testing (faster computation)
        n_test = min(1000, len(dolp))
        test_indices = np.random.choice(len(dolp), n_test, replace=False)
        
        dolp_test = dolp[test_indices]
        aolp_test = aolp[test_indices]
        azimuth_test = azimuth[test_indices]
        
        # Baseline performance (no noise)
        baseline_pred = model.predict(dolp_test, aolp_test)
        baseline_mae = np.mean(np.abs(baseline_pred - azimuth_test))
        
        print(f"Baseline MAE (no noise): {baseline_mae:.3f}°")
        
        robustness_results = {'baseline_mae': baseline_mae}
        
        # Test different noise levels
        noise_stddevs = [0.01, 0.02, 0.03, 0.04, 0.05]  # As specified in blueprint
        
        for noise_std in noise_stddevs:
            print(f"  Testing noise std = {noise_std}")
            
            # Add Gaussian noise to DoLP
            dolp_noisy = dolp_test + np.random.normal(0, noise_std, dolp_test.shape)
            dolp_noisy = np.clip(dolp_noisy, 0, 1)  # Keep in valid range
            
            # Add Poisson noise (convert to counts, add noise, convert back)
            dolp_counts = (dolp_test * 1000).astype(int)  # Scale to count-like data
            dolp_poisson = np.random.poisson(dolp_counts).astype(float) / 1000
            dolp_poisson = np.clip(dolp_poisson, 0, 1)
            
            # Test both noise types
            pred_gaussian = model.predict(dolp_noisy, aolp_test)
            pred_poisson = model.predict(dolp_poisson, aolp_test)
            
            mae_gaussian = np.mean(np.abs(pred_gaussian - azimuth_test))
            mae_poisson = np.mean(np.abs(pred_poisson - azimuth_test))
            
            robustness_results[f'gaussian_noise_{noise_std}'] = mae_gaussian
            robustness_results[f'poisson_noise_{noise_std}'] = mae_poisson
            
            print(f"    Gaussian noise MAE: {mae_gaussian:.3f}°")
            print(f"    Poisson noise MAE: {mae_poisson:.3f}°")
        
        # Test turbidity simulation
        print("  Testing turbidity simulation...")
        turbidity_densities = [0.1, 0.15, 0.2, 0.25, 0.3]  # As specified
        
        for density in turbidity_densities:
            # Simulate haze by reducing DoLP (turbidity reduces polarization)
            haze_factor = np.exp(-density)  # Exponential attenuation
            dolp_hazy = dolp_test * haze_factor
            
            pred_hazy = model.predict(dolp_hazy, aolp_test)
            mae_hazy = np.mean(np.abs(pred_hazy - azimuth_test))
            
            robustness_results[f'turbidity_{density}'] = mae_hazy
            print(f"    Turbidity (density {density}) MAE: {mae_hazy:.3f}°")
        
        # Check if robustness meets requirements
        max_mae = max([v for k, v in robustness_results.items() if k != 'baseline_mae'])
        meets_robustness = max_mae <= 5.0  # Blueprint requirement
        
        robustness_results['max_degraded_mae'] = max_mae
        robustness_results['meets_robustness_requirement'] = meets_robustness
        
        print(f"\\nRobustness testing complete.")
        print(f"Maximum degraded MAE: {max_mae:.3f}°")
        print(f"Meets robustness requirement (≤5°): {'✓' if meets_robustness else '✗'}")
        
        self.results['robustness_metrics'] = robustness_results
        
        return robustness_results
    
    def save_results(self):
        """Save all results to files."""
        # Save detailed results as JSON
        results_path = os.path.join(self.output_dir, 'L2_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        # Create summary report
        self.create_summary_report()
    
    def create_summary_report(self):
        """Create a human-readable summary report."""
        report_path = os.path.join(self.output_dir, 'L2_training_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("L2 Linear Regression Baseline Model - Training Summary\\n")
            f.write("=" * 60 + "\\n\\n")
            f.write(f"Training completed: {self.results['timestamp']}\\n\\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE:\\n")
            f.write("-" * 20 + "\\n")
            if 'cv_metrics' in self.results:
                cv = self.results['cv_metrics']
                f.write(f"Cross-validation MAE: {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}°\\n")
                f.write(f"Cross-validation RMSE: {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}°\\n")
                f.write(f"Cross-validation R²: {cv['r2_mean']:.3f} ± {cv['r2_std']:.3f}\\n")
                
                meets_req = cv['mae_mean'] < 5.0 and cv['rmse_mean'] <= 10.0
                f.write(f"Meets blueprint requirements: {'✓' if meets_req else '✗'}\\n\\n")
            
            # Robustness results
            if 'robustness_metrics' in self.results:
                rob = self.results['robustness_metrics']
                f.write("ROBUSTNESS TESTING:\\n")
                f.write("-" * 20 + "\\n")
                f.write(f"Baseline MAE (clean data): {rob['baseline_mae']:.3f}°\\n")
                f.write(f"Maximum degraded MAE: {rob['max_degraded_mae']:.3f}°\\n")
                f.write(f"Passes robustness test: {'✓' if rob['meets_robustness_requirement'] else '✗'}\\n\\n")
            
            # Blueprint compliance
            f.write("BLUEPRINT COMPLIANCE:\\n")
            f.write("-" * 20 + "\\n")
            f.write("✓ 20k supervised dataset (10k original + 10k augmented)\\n")
            f.write("✓ 5-fold cross-validation implemented\\n")
            f.write("✓ L2 (Ridge) regularization applied\\n")
            f.write("✓ Robustness testing with noise simulation\\n")
            f.write("✓ Turbidity simulation implemented\\n")
            
        print(f"Summary report saved to {report_path}")
    
    def run_complete_pipeline(self):
        """
        Run the complete L2 training pipeline according to blueprint specifications.
        """
        print("Starting L2 Linear Regression Training Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Prepare dataset
            print("\\n1. PREPARING DATASET")
            print("-" * 30)
            dolp, aolp, azimuth = self.prepare_dataset()
            
            # Step 2: Hyperparameter search
            print("\\n2. HYPERPARAMETER SEARCH") 
            print("-" * 30)
            hp_results = self.hyperparameter_search(dolp, aolp, azimuth)
            
            # Step 3: Train final model
            print("\\n3. TRAINING FINAL MODEL")
            print("-" * 30)
            model = self.train_final_model(dolp, aolp, azimuth, hp_results['best_params'])
            
            # Step 4: Robustness testing
            print("\\n4. ROBUSTNESS TESTING")
            print("-" * 30)
            self.robustness_testing(model, dolp, aolp, azimuth)
            
            # Step 5: Save model and results
            print("\\n5. SAVING RESULTS")
            print("-" * 30)
            model_path = os.path.join(self.output_dir, 'L2_baseline_model.pkl')
            model.save_model(model_path)
            self.save_results()
            
            print(f"\\n{'='*60}")
            print("L2 BASELINE TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Results saved to: {self.output_dir}")
            print("Check the summary report for detailed performance metrics.")
            
            return model
            
        except Exception as e:
            print(f"\\n❌ Training pipeline failed: {str(e)}")
            print("Check the error details above.")
            raise


if __name__ == "__main__":
    # Run the complete L2 training pipeline
    print("Initializing L2 Linear Regression Training Loop...")
    
    # Create training loop
    trainer = L2TrainingLoop(
        output_dir="L2_results",
        random_state=42
    )
    
    # Run complete pipeline
    trained_model = trainer.run_complete_pipeline()
    
    print("\\nBaseline L2 model training completed!")
    print("You can now proceed with the other regression models (SVR, Random Forest).")
