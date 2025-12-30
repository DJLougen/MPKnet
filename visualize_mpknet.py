#!/usr/bin/env python3
"""
MPKNet (SGD Version) Architecture Visualization
Updated to reflect CellPopDownsample3x3 stems
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up figure
fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'input': '#E8E8E8',
    'prempk': '#DDA0DD',   # Plum - preprocessing
    'cellpop': '#FFD700',  # Gold - CellPop stems
    'magno': '#87CEEB',    # Sky blue - fast/global
    'parvo': '#98FB98',    # Pale green - detail
    'konio': '#FFA07A',    # Light salmon - modulatory
    'fuse': '#E6E6FA',     # Lavender
    'head': '#B0C4DE',     # Light steel blue
}

def draw_block(ax, x, y, w, h, label, sublabel, color, fontsize=9):
    """Draw a rounded rectangle block with label."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                fontsize=7, style='italic', color='#555555')

def draw_arrow(ax, start, end, color='#333333', style='->', lw=1.5):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# Title
ax.text(8, 10.5, 'MPKNet Architecture (SGD Version)', ha='center', va='center',
        fontsize=16, fontweight='bold')
ax.text(8, 10.1, 'Magno-Parvo-Konio Visual Processing with CellPop Stems', ha='center', va='center',
        fontsize=10, style='italic', color='#666666')

# Input
draw_block(ax, 0.3, 4.5, 1.3, 1.2, 'Input', '3×H×W', colors['input'])

# PreMPK (LGN-like preprocessing)
draw_block(ax, 2.2, 4.5, 1.6, 1.2, 'PreMPK', 'LGN Filters', colors['prempk'])
ax.text(3, 3.9, 'HPF (Parvo)\nLPF (Magno)', ha='center', fontsize=6, color='#666666')

# Arrow from input to PreMPK
draw_arrow(ax, (1.55, 5.1), (2.25, 5.1))

# === CellPop Stems (3x3 structured downsampling) ===
# Parvo CellPop stem
draw_block(ax, 4.5, 7.2, 1.8, 1, 'CellPop', 'P-stem: 24ch', colors['cellpop'])
ax.text(5.4, 6.6, 'pixel_unshuffle(3)\n→ 1×1 group conv', ha='center', fontsize=5, color='#666666')

# Magno CellPop stem
draw_block(ax, 4.5, 4.5, 1.8, 1, 'CellPop', 'M-stem: 24ch', colors['cellpop'])
ax.text(5.4, 3.9, 'pixel_unshuffle(3)\n→ 1×1 group conv', ha='center', fontsize=5, color='#666666')

# Konio CellPop stem
draw_block(ax, 4.5, 1.8, 1.8, 1, 'CellPop', 'K-stem: 12ch', colors['cellpop'])
ax.text(5.4, 1.2, 'pixel_unshuffle(3)\n→ 1×1 group conv', ha='center', fontsize=5, color='#666666')

# PreMPK to CellPop stems
draw_arrow(ax, (3.75, 5.5), (4.55, 7.5))  # To Parvo
ax.text(3.8, 6.6, 'HPF', fontsize=7, color=colors['parvo'], fontweight='bold')

draw_arrow(ax, (3.75, 5.0), (4.55, 5.0))  # To Magno
ax.text(3.9, 4.6, 'LPF', fontsize=7, color=colors['magno'], fontweight='bold')

# Raw input to Konio (bypasses PreMPK)
ax.annotate('', xy=(4.55, 2.3), xytext=(2.2, 4.6),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5,
                           connectionstyle='arc3,rad=-0.3'))
ax.text(2.8, 3.0, 'raw RGB', fontsize=6, color='#666666', rotation=-40)

# === Main Pathways ===
# Parvo pathway (top)
draw_block(ax, 7, 7.2, 2.4, 1, 'Parvo', '4×4→3×3→2×2→2×2', colors['parvo'])
ax.text(8.2, 6.5, '64ch, 4 layers\nSmall RF, detail', ha='center', fontsize=6, color='#555555')

# Magno pathway (middle)
draw_block(ax, 7, 4.5, 2.4, 1, 'Magno', '7×7/s1 → 9×9/s2', colors['magno'])
ax.text(8.2, 3.8, '64ch, 2 layers\nLarge RF, gist', ha='center', fontsize=6, color='#555555')

# Konio pathway (bottom)
draw_block(ax, 7, 1.8, 2.4, 1, 'Konio', '5×5/s1 × 3', colors['konio'])
ax.text(8.2, 1.1, '16ch, 3 layers\nContext/modulation', ha='center', fontsize=6, color='#555555')

# CellPop to Pathways
draw_arrow(ax, (6.25, 7.7), (7.05, 7.7))  # Parvo
draw_arrow(ax, (6.25, 5.0), (7.05, 5.0))  # Magno
draw_arrow(ax, (6.25, 2.3), (7.05, 2.3))  # Konio

# === Fusion ===
draw_block(ax, 10.2, 4.5, 1.6, 1.2, 'Fuse', '1×1 Conv', colors['fuse'])
ax.text(11, 3.9, 'cat(P,M,K)→128ch', ha='center', fontsize=6, color='#555555')

# Streams to fusion with interpolation
draw_arrow(ax, (9.35, 7.7), (10.25, 5.5))
draw_arrow(ax, (9.35, 5.0), (10.25, 5.1))
draw_arrow(ax, (9.35, 2.3), (10.25, 4.7))

ax.text(9.6, 6.8, '↑interp', fontsize=6, color='#888888')
ax.text(9.6, 3.5, '↑interp', fontsize=6, color='#888888')

# === Classifier Head ===
draw_block(ax, 12.5, 4.5, 1.4, 1.2, 'Head', 'GAP→FC', colors['head'])
draw_arrow(ax, (11.75, 5.1), (12.55, 5.1))

# Output
ax.text(14.3, 5.1, 'logits', ha='center', fontsize=9)
draw_arrow(ax, (13.85, 5.1), (14.1, 5.1))

# === Info boxes ===
# Model summary
param_text = """Model Summary (SGD Version):
• Total: ~0.6M params
• SGD + warmup + cosine LR
• SWA (Stochastic Weight Avg)
• EMA (Exponential Moving Avg)
• Mixup α=0.2
• Label smoothing 0.1"""
ax.text(0.5, 1.5, param_text, fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# CellPop explanation
cellpop_text = """CellPopDownsample3x3:
• pixel_unshuffle(s) → [B, s²·C, H/s, W/s]
• 1×1 grouped conv (9→k per channel)
• 1×1 pointwise mix → output channels
• Efficient structured downsampling"""
ax.text(0.5, 8.5, cellpop_text, fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor=colors['cellpop'], alpha=0.5))

# Biological correspondence
bio_text = """Biological Basis:
• PreMPK → LGN relay filtering
• CellPop → Retinal ganglion sampling
• Magno → M-pathway (motion, gist)
• Parvo → P-pathway (color, detail)
• Konio → K-pathway (context relay)"""
ax.text(12.5, 8.5, bio_text, fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Training features
train_text = """Training Features:
• PReLU activations
• BatchNorm after each conv
• Cosine LR with warmup
• SWA after 52.5% epochs"""
ax.text(12.5, 1.5, train_text, fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Legend
legend_data = [
    (colors['cellpop'], 'CellPop: Structured downsample'),
    (colors['magno'], 'Magno: Large RF, global gist'),
    (colors['parvo'], 'Parvo: Small RF, fine detail'),
    (colors['konio'], 'Konio: Context relay'),
]

for i, (color, label) in enumerate(legend_data):
    y_pos = 9.8 - i * 0.35
    rect = plt.Rectangle((6.5, y_pos - 0.1), 0.3, 0.25, facecolor=color, edgecolor='#333')
    ax.add_patch(rect)
    ax.text(6.9, y_pos, label, fontsize=7, va='center')

plt.tight_layout()
plt.savefig('mpknet_architecture.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('mpknet_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: mpknet_architecture.png and mpknet_architecture.pdf")
