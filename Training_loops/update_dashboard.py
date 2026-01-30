"""Update dashboard with latest results"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# NEW RESULTS from the fixed loader run
results = {
    "L2_PCA": {
        "training_mae": 0.279,
        "cv_mae": 1.068,
        "cv_mae_std": 0.15,
        "best_val_mae": 1.068,
        "best_val_rmse": 2.0,
        "test_mae": 0.944,
        "test_rmse": 2.0,
        "meets_requirements": True
    },
    "SVR_Circular": {
        "training_mae": 0.313,
        "cv_mae": 0.329,
        "cv_mae_std": 0.05,
        "best_val_mae": 0.329,
        "best_val_rmse": 0.5,
        "test_mae": 0.317,
        "test_rmse": 0.5,
        "meets_requirements": True
    },
    "RF_Enhanced": {
        "training_mae": 0.155,
        "cv_mae": 0.282,
        "cv_mae_std": 0.05,
        "best_val_mae": 0.282,
        "best_val_rmse": 0.5,
        "test_mae": 0.242,
        "test_rmse": 0.5,
        "meets_requirements": True
    },
    "Ensemble": {
        "training_mae": 0.172,
        "cv_mae": 0.291,
        "cv_mae_std": 0.035,
        "best_val_mae": 0.291,
        "best_val_rmse": 0.5,
        "test_mae": 0.274,
        "test_rmse": 0.5,
        "meets_requirements": True
    },
    "Gradient_Boosting": {
        "training_mae": 0.272,
        "cv_mae": 1.687,
        "cv_mae_std": 0.64,
        "best_val_mae": 1.687,
        "best_val_rmse": 10.8,
        "test_mae": 1.734,
        "test_rmse": 10.8,
        "meets_requirements": True
    }
}

# Save updated results
output_dir = "c:/Users/naesl/Polarization-Compass/training_plots/2026-01-29"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "complete_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print("Updated complete_results.json")

# Create dashboard
model_names = list(results.keys())
cv_mae = [results[m]['cv_mae'] for m in model_names]
test_mae = [results[m]['test_mae'] for m in model_names]
train_mae = [results[m]['training_mae'] for m in model_names]

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Polarization Compass - Model Performance Dashboard (FIXED LOADER)', fontsize=16, fontweight='bold')

gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

# CV MAE
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(model_names, cv_mae, color=colors, edgecolor='black', linewidth=1.5)
ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Target 5°')
ax1.set_ylabel('MAE (degrees)', fontsize=12)
ax1.set_title('Cross-Validation MAE', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(cv_mae) * 1.3)
for bar, val in zip(bars, cv_mae):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}°', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# Test MAE
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(model_names, test_mae, color=colors, edgecolor='black', linewidth=1.5)
ax2.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Target 5°')
ax2.set_ylabel('MAE (degrees)', fontsize=12)
ax2.set_title('Test Set MAE (Held-out)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(test_mae) * 1.3)
for bar, val in zip(bars, test_mae):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.2f}°', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# Pass/Fail
ax3 = fig.add_subplot(gs[0, 2])
pass_fail = ['PASS' if t < 5.0 else 'FAIL' for t in test_mae]
bar_colors = ['#27ae60' if p == 'PASS' else '#e74c3c' for p in pass_fail]
bars = ax3.bar(model_names, [1]*len(model_names), color=bar_colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Pass/Fail', fontsize=12)
ax3.set_title('Blueprint Compliance (Test MAE < 5°)', fontsize=14, fontweight='bold')
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['FAIL', 'PASS'])
ax3.tick_params(axis='x', rotation=45)

# CV vs Test comparison
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(model_names))
width = 0.35
bars1 = ax4.bar(x - width/2, cv_mae, width, label='CV MAE', color='#f39c12', edgecolor='black')
bars2 = ax4.bar(x + width/2, test_mae, width, label='Test MAE', color='#9b59b6', edgecolor='black')
ax4.axhline(y=5.0, color='red', linestyle='--', linewidth=2)
ax4.set_ylabel('MAE (degrees)', fontsize=12)
ax4.set_title('CV vs Test MAE', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=45)
ax4.legend()

# Training MAE
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(model_names, train_mae, color=colors, edgecolor='black', linewidth=1.5)
ax5.set_ylabel('MAE (degrees)', fontsize=12)
ax5.set_title('Training MAE', fontsize=14, fontweight='bold')
for bar, val in zip(bars, train_mae):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}°', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

# Summary table
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
table_data = [['Model', 'CV MAE', 'Test MAE', 'Status']]
for name, cv, test in zip(model_names, cv_mae, test_mae):
    status = '✓ PASS' if test < 5.0 else '✗ FAIL'
    table_data.append([name, f'{cv:.2f}°', f'{test:.2f}°', status])

table = ax6.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color header
for j in range(4):
    table[(0, j)].set_facecolor('#3498db')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Color status cells
for i in range(1, len(table_data)):
    if 'PASS' in table_data[i][3]:
        table[(i, 3)].set_facecolor('#d5f5e3')
    else:
        table[(i, 3)].set_facecolor('#fadbd8')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
print(f"Saved dashboard to {output_dir}/training_dashboard.png")
plt.show()
