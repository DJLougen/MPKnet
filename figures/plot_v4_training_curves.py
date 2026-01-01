"""Plot V4 training curves for CIFAR-100, Kvasir, STL-10"""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
cifar100_df = pd.read_csv('cifar100_v4_k3.csv')
kvasir_df = pd.read_csv('v4_kvasir.csv')
stl10_df = pd.read_csv('v4_stl10.csv')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Pastel colors
pastel_blue = '#A8D5E5'    # soft sky blue
pastel_green = '#B8D4B8'   # sage green
pastel_rose = '#E8B4B8'    # dusty rose

# Plot curves (these are test accuracy)
ax.plot(cifar100_df['Step'].values, cifar100_df['Value'].values,
        color=pastel_blue, linewidth=2, label='CIFAR-100 (58.8%)')
ax.plot(kvasir_df['Step'].values, kvasir_df['Value'].values,
        color=pastel_green, linewidth=2, label='Kvasir (89.2%)')
ax.plot(stl10_df['Step'].values, stl10_df['Value'].values,
        color=pastel_rose, linewidth=2, label='STL-10 (71.7%)')

# Styling
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('MPKx Test Accuracy (0.21M params, no augmentation)', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('v4_training_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('v4_training_curves.pdf', bbox_inches='tight')
print("Saved v4_training_curves.png and .pdf")
