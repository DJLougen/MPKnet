#!/usr/bin/env python3
"""Generate a professional MPKNet architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'input': '#E8E8E8',
    'prempk': '#B8D4E3',
    'parvo': '#98D898',  # Green for P
    'magno': '#E89898',  # Red for M
    'konio': '#D8B8E8',  # Purple for K
    'fusion': '#F5DEB3',
    'head': '#FFE4B5',
    'arrow': '#404040',
}

def draw_box(ax, x, y, w, h, color, label, sublabel=None, fontsize=10):
    """Draw a rounded box with label."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='#404040', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y + (0.1 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold')
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha='center', va='center',
                fontsize=fontsize-2, style='italic', color='#606060')

def draw_arrow(ax, start, end, color='#404040'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Title
ax.text(6, 9.5, 'MPKNet Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold')

# Input
draw_box(ax, 6, 8.5, 2.5, 0.6, colors['input'], 'Input', '3×H×W')

# PreMPK
draw_box(ax, 6, 7.3, 3, 0.7, colors['prempk'], 'PreMPK', 'HPF→P, LPF→M')

# Arrows from input to PreMPK
draw_arrow(ax, (6, 8.2), (6, 7.7))

# CellPop stems (3 parallel)
stem_y = 6.0
stem_positions = [3.5, 6, 8.5]
stem_labels = ['P-stem', 'M-stem', 'K-stem']
stem_colors = [colors['parvo'], colors['magno'], colors['konio']]

for i, (x, label, color) in enumerate(zip(stem_positions, stem_labels, stem_colors)):
    draw_box(ax, x, stem_y, 2, 0.6, color, 'CellPop', label)

# Arrows from PreMPK to stems
for x in stem_positions:
    draw_arrow(ax, (6, 6.9), (x, 6.35))

# M/P/K Pathways
pathway_y = 4.5
pathway_info = [
    (3.5, 'Parvo', '3×3→3×3\n4 layers', colors['parvo']),
    (6, 'Magno', '7×7→5×5\n2 layers', colors['magno']),
    (8.5, 'Konio', '5×5\n3 layers', colors['konio']),
]

for x, label, sublabel, color in pathway_info:
    draw_box(ax, x, pathway_y, 2, 1.0, color, label, sublabel)

# Arrows from stems to pathways
for x in stem_positions:
    draw_arrow(ax, (x, 5.7), (x, 5.05))

# K-Gate
draw_box(ax, 8.5, 3.2, 1.8, 0.5, colors['konio'], 'K-Gate (σ)')
draw_arrow(ax, (8.5, 3.95), (8.5, 3.5))

# Dotted lines from K-Gate to P and M (modulation)
ax.annotate('', xy=(3.5, 3.5), xytext=(7.6, 3.2),
            arrowprops=dict(arrowstyle='->', color=colors['konio'],
                          lw=1.5, linestyle='dashed'))
ax.annotate('', xy=(6, 3.5), xytext=(7.6, 3.2),
            arrowprops=dict(arrowstyle='->', color=colors['konio'],
                          lw=1.5, linestyle='dashed'))

# Fusion
draw_box(ax, 6, 2.3, 4, 0.7, colors['fusion'], 'Fusion', '1×1 conv')

# Arrows to fusion
draw_arrow(ax, (3.5, 3.95), (4.5, 2.7))
draw_arrow(ax, (6, 3.95), (6, 2.7))

# Head
draw_box(ax, 6, 1.2, 3, 0.7, colors['head'], 'Head', 'GAP → FC → logits')
draw_arrow(ax, (6, 1.95), (6, 1.6))

# Legend
legend_y = 0.3
legend_items = [
    (1.5, 'P: Fine detail, color', colors['parvo']),
    (4.5, 'M: Motion, global gist', colors['magno']),
    (7.5, 'K: Context modulation', colors['konio']),
]
for x, label, color in legend_items:
    rect = FancyBboxPatch((x - 0.3, legend_y - 0.15), 0.3, 0.3,
                          boxstyle="round,pad=0.01", facecolor=color,
                          edgecolor='#404040', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + 0.2, legend_y, label, ha='left', va='center', fontsize=9)

# Add annotation for dashed lines
ax.text(10.5, 3.2, 'modulation', ha='left', va='center',
        fontsize=8, style='italic', color='#606060')

plt.tight_layout()
plt.savefig('figures/mpknet_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figures/mpknet_architecture.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved figures/mpknet_architecture.png and .svg")
