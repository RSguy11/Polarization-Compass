"""Generate fixed training dashboard chart"""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-darkgrid")

model_names = ["L2_PCA", "SVR_Circular", "RF_Enhanced", "Ensemble", "Gradient_Boosting"]
cv_mae = [5.12, 1.40, 1.36, 1.57, 5.83]
test_mae = [8.14, 2.46, 1.87, 3.79, 4.33]
train_mae = [5.04, 0.39, 0.43, 0.66, 1.07]
meets_req = [False, True, True, True, True]

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

# CV MAE
ax1 = fig.add_subplot(gs[0, 0])
colors1 = ["#e74c3c" if not m else "#2ecc71" for m in meets_req]
ax1.bar(range(len(model_names)), cv_mae, color=colors1, alpha=0.8, edgecolor="black")
ax1.axhline(y=5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Target 5°")
ax1.set_ylabel("MAE (degrees)", fontweight="bold")
ax1.set_title("Cross-Validation MAE", fontweight="bold", fontsize=12)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
for i, v in enumerate(cv_mae):
    ax1.text(i, v + 0.2, f"{v:.2f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.legend()

# Test MAE
ax2 = fig.add_subplot(gs[0, 1])
colors2 = ["#e74c3c" if not m else "#2ecc71" for m in meets_req]
ax2.bar(range(len(model_names)), test_mae, color=colors2, alpha=0.8, edgecolor="black")
ax2.axhline(y=5.0, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Target 5°")
ax2.set_ylabel("MAE (degrees)", fontweight="bold")
ax2.set_title("Test Set MAE (Held-out)", fontweight="bold", fontsize=12)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
for i, v in enumerate(test_mae):
    ax2.text(i, v + 0.2, f"{v:.2f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.legend()

# Pass/Fail
ax3 = fig.add_subplot(gs[0, 2])
colors3 = ["#2ecc71" if m else "#e74c3c" for m in meets_req]
ax3.bar(range(len(model_names)), [1 if m else 0 for m in meets_req], color=colors3, alpha=0.8, edgecolor="black")
ax3.set_ylabel("Pass/Fail", fontweight="bold")
ax3.set_title("Blueprint Compliance (Test MAE < 5°)", fontweight="bold", fontsize=12)
ax3.set_ylim([0, 1.3])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["FAIL", "PASS"])
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)

# CV vs Test comparison
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(model_names))
width = 0.35
ax4.bar(x - width/2, cv_mae, width, label="CV MAE", color="#f39c12", alpha=0.8, edgecolor="black")
ax4.bar(x + width/2, test_mae, width, label="Test MAE", color="#9b59b6", alpha=0.8, edgecolor="black")
ax4.axhline(y=5.0, color="red", linestyle="--", linewidth=2, alpha=0.7)
ax4.set_ylabel("MAE (degrees)", fontweight="bold")
ax4.set_title("CV vs Test MAE", fontweight="bold", fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
ax4.legend()

# Training MAE
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(len(model_names)), train_mae, color="#3498db", alpha=0.8, edgecolor="black")
ax5.set_ylabel("MAE (degrees)", fontweight="bold")
ax5.set_title("Training MAE", fontweight="bold", fontsize=12)
ax5.set_xticks(range(len(model_names)))
ax5.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
for i, v in enumerate(train_mae):
    ax5.text(i, v + 0.05, f"{v:.2f}°", ha="center", va="bottom", fontsize=9)

# Summary table as text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
table_data = [["Model", "CV MAE", "Test MAE", "Status"]]
for i, name in enumerate(model_names):
    status = "✓ PASS" if meets_req[i] else "✗ FAIL"
    table_data.append([name, f"{cv_mae[i]:.2f}°", f"{test_mae[i]:.2f}°", status])

table = ax6.table(cellText=table_data, loc="center", cellLoc="center", 
                  colWidths=[0.35, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
# Color header
for j in range(4):
    table[(0, j)].set_facecolor("#3498db")
    table[(0, j)].set_text_props(color="white", fontweight="bold")
# Color results
for i in range(1, 6):
    if meets_req[i-1]:
        table[(i, 3)].set_facecolor("#d5f5e3")
    else:
        table[(i, 3)].set_facecolor("#fadbd8")

plt.suptitle("Polarization Compass - Model Performance Dashboard", fontsize=16, fontweight="bold", y=0.98)

os.makedirs("training_plots/2026-01-29", exist_ok=True)
plt.savefig("training_plots/2026-01-29/training_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart saved to training_plots/2026-01-29/training_dashboard.png")
