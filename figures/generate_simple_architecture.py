"""
Generate architecture diagram matching Keynote style with strides.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
RED = '#C0392B'
NAVY = '#2E4053'
GREEN = '#58D68D'
BLUE = '#5DADE2'
GOLD = '#F4D03F'
PURPLE = '#AF7AC5'
GRAY = '#D5D8DC'

def add_box(ax, x, y, w, h, color, text, fontsize=11, textcolor='black'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                         facecolor=color, edgecolor='none')
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='medium', color=textcolor)

def add_arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Image placeholder
ax.add_patch(plt.Rectangle((0.5, 3.2), 1.2, 1.2, facecolor='black'))
ax.text(1.1, 2.7, 'Image', ha='center', va='top', fontsize=12)

# Arrow from Image to Retinal
add_arrow(ax, (1.7, 3.8), (2.5, 3.8))

# Retinal Preprocessing
add_box(ax, 2.5, 3, 1.8, 1.6, RED, 'Retinal\nPreprocessing\n(Center surround)', fontsize=10, textcolor='white')

# Arrows from Retinal to Right/Left
add_arrow(ax, (4.3, 4.0), (5.0, 4.8))
add_arrow(ax, (4.3, 3.6), (5.0, 2.8))

# Right eye
add_box(ax, 5.0, 4.5, 1.0, 0.7, RED, 'Right', fontsize=11, textcolor='white')

# Left eye
add_box(ax, 5.0, 2.4, 1.0, 0.7, NAVY, 'Left', fontsize=11, textcolor='white')

# Arrows from Right to M/P/K
add_arrow(ax, (6.0, 5.0), (7.2, 6.2))
add_arrow(ax, (6.0, 4.8), (7.2, 5.0))
add_arrow(ax, (6.0, 4.6), (7.2, 3.9))

# Arrows from Left to M/P/K
add_arrow(ax, (6.0, 2.6), (7.2, 1.4))
add_arrow(ax, (6.0, 2.8), (7.2, 2.6))
add_arrow(ax, (6.0, 3.0), (7.2, 3.6))

# Right eye pathways (top group) - with strides
add_box(ax, 7.2, 6.0, 1.3, 0.6, GREEN, 'Magno (s=3)', fontsize=10)
add_box(ax, 7.2, 4.8, 1.3, 0.6, BLUE, 'Parvo (s=1)', fontsize=10)
add_box(ax, 7.2, 3.6, 1.3, 0.6, GOLD, 'Konio (s=2)', fontsize=10)

# Left eye pathways (bottom group) - with strides
add_box(ax, 7.2, 2.4, 1.3, 0.6, GREEN, 'Magno (s=3)', fontsize=10)
add_box(ax, 7.2, 1.2, 1.3, 0.6, BLUE, 'Parvo (s=1)', fontsize=10)
add_box(ax, 7.2, 0.0, 1.3, 0.6, GOLD, 'Konio (s=2)', fontsize=10)

# K-Gate labels
ax.text(8.7, 4.9, 'K-Gate', fontsize=9, color='gray', style='italic')
ax.text(8.7, 1.3, 'K-Gate', fontsize=9, color='gray', style='italic')

# Dashed lines for K-gating
ax.plot([8.5, 8.5], [3.9, 6.3], 'k--', lw=1, alpha=0.5)
ax.plot([8.5, 8.5], [0.3, 2.7], 'k--', lw=1, alpha=0.5)

# Arrows to V1 Fusion
add_arrow(ax, (8.5, 6.3), (9.5, 4.2))
add_arrow(ax, (8.5, 5.1), (9.5, 4.0))
add_arrow(ax, (8.5, 2.7), (9.5, 3.6))
add_arrow(ax, (8.5, 1.5), (9.5, 3.4))

# V1 Fusion
add_box(ax, 9.5, 3.2, 1.2, 1.2, PURPLE, 'V1\nFusion', fontsize=11, textcolor='white')

# Arrow to GAP
add_arrow(ax, (10.7, 3.8), (11.2, 3.8))

# GAP
add_box(ax, 11.2, 3.4, 0.8, 0.8, GRAY, 'GAP', fontsize=10)

# Arrow to FC
add_arrow(ax, (12.0, 3.8), (12.5, 3.8))

# FC -> Output
add_box(ax, 12.5, 3.4, 1.0, 0.8, GRAY, 'FC\nOutput', fontsize=10)

plt.tight_layout()
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_simple_diagram.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_simple_diagram.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved mpkx_simple_diagram.png and .pdf")
