"""Training Visualization & Plotting"""

import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)


def create_training_plots(results, training_history, output_dir):
    """Create comprehensive training visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    model_names = []
    train_mae = []
    best_val_mae = []
    best_val_rmse = []
    test_mae = []
    test_rmse = []
    cv_mae = []
    
    for name, result in results.items():
        if 'error' not in result:
            model_names.append(name)
            train_mae.append(result['training_mae'])
            best_val_mae.append(result['best_val_mae'])
            best_val_rmse.append(result['best_val_rmse'])
            test_mae.append(result['test_mae'])
            test_rmse.append(result['test_rmse'])
            cv_mae.append(result['cv_mae'])
    
    if not model_names:
        print(" No successful results to plot")
        return
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1], hspace=0.35, wspace=0.35)
    
    ax1 = fig.add_subplot(gs[:, 0])
    
    if training_history:
        for model_name in model_names:
            if model_name in training_history:
                history = training_history[model_name]
                # Skip if sample_sizes and errors have different lengths (e.g., Ensemble)
                if len(history.get('sample_sizes', [])) != len(history.get('train_errors', [])):
                    continue
                if len(history['sample_sizes']) == 0:
                    continue
                ax1.plot(history['sample_sizes'], history['train_errors'], 
                        'o-', label=f'{model_name} (Train)', linewidth=2, markersize=6)
                ax1.plot(history['sample_sizes'], history['val_errors'], 
                        's--', label=f'{model_name} (Val)', linewidth=2, markersize=6, alpha=0.7)
                best_idx = history['sample_sizes'].index(history['best_sample_size'])
                ax1.plot(history['best_sample_size'], history['val_errors'][best_idx], 
                        '*', markersize=20, color='gold', markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=10)
        
        ax1.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='Target (5° MAE)', alpha=0.7)
        ax1.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
        ax1.set_title('Learning Curves: Error vs Training Set Size', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, 'Learning curves require\nincremental training', 
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.set_title('Learning Curves', fontsize=13, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    ax7 = fig.add_subplot(gs[1, 3])
    
    ax2.bar(range(len(model_names)), cv_mae, color='#3498db', alpha=0.8, edgecolor='black')
    ax2.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax2.set_title('Cross-Validation MAE', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(cv_mae):
        ax2.text(i, v + 0.2, f'{v:.2f}°', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax3.bar(range(len(model_names)), test_mae, color='#9b59b6', alpha=0.8, edgecolor='black')
    ax3.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax3.set_title('Test Set MAE (Held-out)', fontweight='bold', fontsize=11)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_mae):
        ax3.text(i, v + 0.2, f'{v:.2f}°', ha='center', va='bottom', fontsize=8, fontweighht='bold')
    
    meets_req = [results[name]['meets_requirements'] for name in model_names]
    colors = ['#2ecc71' if met else '#e74c3c' for met in meets_req]
    ax4.bar(range(len(model_names)), [1 if met else 0 for met in meets_req], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Meets Requirements', fontweight='bold', fontsize=10)
    ax4.set_title('Blueprint Compliance\n(Test MAE < 5°)', fontweight='bold', fontsize=11)
    ax4.set_ylim([0, 1.2])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    
    ax5.bar(range(len(model_names)), train_mae, color='#2ecc71', alpha=0.8, edgecolor='black')
    ax5.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax5.set_title('Training MAE', fontweight='bold', fontsize=11)
    ax5.set_xticks(range(len(model_names)))
    ax5.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate(train_mae):
        ax5.text(i, v + 0.1, f'{v:.2f}°', ha='center', va='bottom', fontsize=8)
    
    # Show test RMSE
    ax6.bar(range(len(model_names)), test_rmse, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax6.set_ylabel('RMSE (degrees)', fontweight='bold', fontsize=10)
    ax6.set_title('Test Set RMSE', fontweight='bold', fontsize=11)
    ax6.set_xticks(range(len(model_names)))
    ax6.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(test_rmse):
        ax6.text(i, v + 0.2, f'{v:.2f}°', ha='center', va='bottom', fontsize=8)
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    ax7.bar(x_pos - width/2, cv_mae, width, label='CV', color='#f39c12', alpha=0.8, edgecolor='black')
    ax7.bar(x_pos + width/2, test_mae, width, label='Test', color='#9b59b6', alpha=0.8, edgecolor='black')
    ax7.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
    ax7.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax7.set_title('CV vs Test MAE', fontweight='bold', fontsize=11)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(model_names, rotation=25, ha='right', fontsize=8)
    ax7.legend(fontsize=8)
    ax7.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Dashboard: Learning Curves & Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training dashboard")
    
    print(f"\nAll plots saved to: {output_dir}")
