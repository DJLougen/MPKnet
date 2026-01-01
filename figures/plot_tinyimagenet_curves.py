"""Plot TinyImageNet training curves for MPKx V4"""
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_df = pd.read_csv('mpkx_tinyimagenetTrain.csv')
test_df = pd.read_csv('mpkx_tinyimagenettTest.csv')

epochs = train_df['Step'].values
train_acc = train_df['Value'].values
test_acc = test_df['Value'].values

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot curves
ax.plot(epochs, train_acc, color='#E57373', linewidth=2, label='Train')
ax.plot(epochs, test_acc, color='#64B5F6', linewidth=2, label='Test')

# Styling
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('MPKx V4 on TinyImageNet-200 (0.21M params, no augmentation)', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

# Add final accuracy annotation
final_train = train_acc[-1]
final_test = test_acc[-1]
gap = final_train - final_test
ax.annotate(f'Final: Train {final_train:.1f}%, Test {final_test:.1f}%\nGap: {gap:.1f}%',
            xy=(95, final_test), xytext=(70, 15),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('tinyimagenet_training_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('tinyimagenet_training_curves.pdf', bbox_inches='tight')
print("Saved tinyimagenet_training_curves.png and .pdf")
