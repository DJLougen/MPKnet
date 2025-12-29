#!/usr/bin/env python3
"""Generate a professional BinocularMPKNet architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.set_aspect('equal')
ax.axis('off')

# Colors
colors = {
    'input': '#E8E8E8',
    'stereo': '#D4E8F0',
    'prempk': '#B8D4E3',
    'parvo': '#98D898',  # Green for P
    'magno': '#E89898',  # Red for M
    'konio': '#D8B8E8',  # Purple for K
    'ocular': '#FFD699',  # Orange for ocular dominance
    'fusion': '#F5DEB3',
    'head': '#FFE4B5',
    'arrow': '#404040',
    'left_eye': '#6CA6CD',  # Blue for left eye
    'right_eye': '#CD6C6C',  # Red for right eye
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

def draw_arrow(ax, start, end, color='#404040', style='-'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, linestyle=style))

# Title
ax.text(7, 10.5, 'BinocularMPKNet Architecture', ha='center', va='center',
        fontsize=16, fontweight='bold')
ax.text(7, 10.1, '0.14M parameters', ha='center', va='center',
        fontsize=11, style='italic', color='#606060')

# Input
draw_box(ax, 7, 9.3, 2.5, 0.6, colors['input'], 'Input', '3xHxW')

# Stereo Disparity
draw_box(ax, 7, 8.3, 3, 0.6, colors['stereo'], 'StereoDisparity', 'horizontal shift')
draw_arrow(ax, (7, 9.0), (7, 8.65))

# Split to two eyes
draw_box(ax, 4.5, 7.3, 2, 0.5, colors['left_eye'], 'Left Eye')
draw_box(ax, 9.5, 7.3, 2, 0.5, colors['right_eye'], 'Right Eye')
draw_arrow(ax, (5.8, 8.0), (4.5, 7.6))
draw_arrow(ax, (8.2, 8.0), (9.5, 7.6))

# BinocularPreMPK (two instances)
draw_box(ax, 4.5, 6.3, 2.5, 0.6, colors['prempk'], 'PreMPK L', 'P_L, M_L')
draw_box(ax, 9.5, 6.3, 2.5, 0.6, colors['prempk'], 'PreMPK R', 'P_R, M_R')
draw_arrow(ax, (4.5, 7.0), (4.5, 6.65))
draw_arrow(ax, (9.5, 7.0), (9.5, 6.65))

# Ocular Dominance Conv (the key binocular fusion point)
pathway_y = 4.8
pathway_info = [
    (3, 'P Pathway', '3x3x3', colors['parvo']),
    (7, 'M Pathway', '7x7, 5x5', colors['magno']),
    (11, 'K Pathway', '5x5x2', colors['konio']),
]

for x, label, sublabel, color in pathway_info:
    draw_box(ax, x, pathway_y, 2.2, 1.0, color, label, sublabel)

# Ocular Dominance boxes inside pathways
ocular_y = 5.5
for x in [3, 7, 11]:
    draw_box(ax, x, ocular_y, 1.8, 0.4, colors['ocular'], 'OcularDom', fontsize=8)

# Arrows from eyes to pathways (showing binocular input)
# Left eye inputs
draw_arrow(ax, (3.8, 6.0), (3, 5.75), color=colors['left_eye'])
draw_arrow(ax, (4.0, 6.0), (7, 5.75), color=colors['left_eye'])
draw_arrow(ax, (4.2, 6.0), (10.2, 5.75), color=colors['left_eye'])

# Right eye inputs
draw_arrow(ax, (8.8, 6.0), (3.8, 5.75), color=colors['right_eye'])
draw_arrow(ax, (9.2, 6.0), (7.8, 5.75), color=colors['right_eye'])
draw_arrow(ax, (10.2, 6.0), (11, 5.75), color=colors['right_eye'])

# K-Gate
draw_box(ax, 11, 3.5, 1.8, 0.5, colors['konio'], 'K-Gate')
draw_arrow(ax, (11, 4.25), (11, 3.8))

# Dotted lines from K-Gate to P and M (modulation)
ax.annotate('', xy=(3, 3.8), xytext=(10.1, 3.5),
            arrowprops=dict(arrowstyle='->', color=colors['konio'],
                          lw=1.5, linestyle='dashed'))
ax.annotate('', xy=(7, 3.8), xytext=(10.1, 3.5),
            arrowprops=dict(arrowstyle='->', color=colors['konio'],
                          lw=1.5, linestyle='dashed'))

# Fusion
draw_box(ax, 7, 2.5, 4, 0.7, colors['fusion'], 'Fusion', '1x1 conv, 96ch')

# Arrows to fusion
draw_arrow(ax, (3, 4.25), (5.5, 2.9))
draw_arrow(ax, (7, 4.25), (7, 2.9))

# Head
draw_box(ax, 7, 1.3, 3, 0.7, colors['head'], 'Head', 'GAP -> FC -> 10')
draw_arrow(ax, (7, 2.1), (7, 1.7))

# Legend - aligned on left
legend_x = 0.5
legend_items = [
    (1.1, 'P: Fine detail', colors['parvo']),
    (0.8, 'M: Motion/gist', colors['magno']),
    (0.5, 'K: Modulation', colors['konio']),
    (0.2, 'Ocular Dom.', colors['ocular']),
]
for y, label, color in legend_items:
    rect = FancyBboxPatch((legend_x, y - 0.1), 0.25, 0.25,
                          boxstyle="round,pad=0.01", facecolor=color,
                          edgecolor='#404040', linewidth=1)
    ax.add_patch(rect)
    ax.text(legend_x + 0.4, y, label, ha='left', va='center', fontsize=9)

# Eye color legend - stacked below
rect_L = FancyBboxPatch((3.5, 0.8), 0.25, 0.25,
                        boxstyle="round,pad=0.01", facecolor=colors['left_eye'],
                        edgecolor='#404040', linewidth=1)
ax.add_patch(rect_L)
ax.text(3.9, 0.9, 'Left eye', ha='left', va='center', fontsize=9)

rect_R = FancyBboxPatch((3.5, 0.4), 0.25, 0.25,
                        boxstyle="round,pad=0.01", facecolor=colors['right_eye'],
                        edgecolor='#404040', linewidth=1)
ax.add_patch(rect_R)
ax.text(3.9, 0.5, 'Right eye', ha='left', va='center', fontsize=9)

# Add annotation
ax.text(12.5, 3.5, 'gating', ha='left', va='center',
        fontsize=8, style='italic', color='#606060')

plt.tight_layout()
plt.savefig('figures/binocular_mpknet_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figures/binocular_mpknet_architecture.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved figures/binocular_mpknet_architecture.png and .svg")
