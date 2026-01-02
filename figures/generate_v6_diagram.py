"""
Generate MPKxV1 (V6) architecture diagram.
Extends MPKx with V1 Simple/Complex pathways.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 20))
ax.set_xlim(0, 18)
ax.set_ylim(0, 20)
ax.axis('off')

# Colors
GRAY = '#E8E8E8'
GREEN = '#A8D5A2'  # P pathway
PINK = '#D4A5C9'   # K pathway
PEACH = '#F5CBA7'  # M pathway
BLUE = '#A9CCE3'   # K-Gate
YELLOW = '#F9E79F' # LGN→V1 Fusion
MAUVE = '#D7BDE2'  # GAP + FC
ORANGE = '#F5B041'  # V1 Simple
TEAL = '#76D7C4'    # V1 Complex
LAVENDER = '#BB8FCE' # V1 Fusion

NAVY = '#2E4053'  # Left eye
DARK_RED = '#922B21'  # Right eye

def add_box(ax, x, y, w, h, color, lines, fontsize=12, textcolor='black'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=color, edgecolor='#666666', linewidth=1)
    ax.add_patch(box)
    total_lines = len(lines)
    for i, (text, fs, style) in enumerate(lines):
        offset = (total_lines - 1) / 2 - i
        ax.text(x, y + offset * 0.35, text, ha='center', va='center',
                fontsize=fs, fontstyle=style, fontweight='bold' if style == 'normal' else 'normal',
                color=textcolor)

def add_arrow(ax, start, end, style='-', color=None):
    if color is None:
        color = '#666666' if style == '-' else '#9B59B6'
    ls = '-' if style == '-' else '--'
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls=ls))

def add_angled_arrow(ax, start, end, go_horizontal_first=True, style='-', color=None):
    """Draw a 90-degree angled arrow (L-shaped path)."""
    if color is None:
        color = '#666666' if style == '-' else '#9B59B6'
    ls = '-' if style == '-' else '--'
    if go_horizontal_first:
        mid = (end[0], start[1])  # go horizontal first, then vertical
    else:
        mid = (start[0], end[1])  # go vertical first, then horizontal
    # Draw the two line segments
    ax.plot([start[0], mid[0]], [start[1], mid[1]], color=color, lw=1.5, ls=ls)
    # Draw arrow for second segment
    ax.annotate('', xy=end, xytext=mid,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls=ls))

# Title
ax.text(9, 19.5, 'MPKxV1 (V6)', fontsize=24, ha='center', va='center', fontweight='bold')
ax.text(9, 19.0, '0.423M | 3x3 | LGN strides 1:2:3 | V1 strides 1:2', fontsize=12, ha='center', va='center',
        fontstyle='italic', color='#666666')

# Section labels
ax.text(1.0, 16.5, 'RETINA', fontsize=11, ha='center', va='center', fontweight='bold', color='#555555', rotation=90)
ax.text(1.0, 13.0, 'LGN', fontsize=11, ha='center', va='center', fontweight='bold', color='#555555', rotation=90)
ax.text(1.0, 5.5, 'V1', fontsize=11, ha='center', va='center', fontweight='bold', color='#555555', rotation=90)

# Input
add_box(ax, 9, 18.3, 2.5, 0.8, GRAY, [('Input', 12, 'normal'), ('96x96', 10, 'italic')])

# Arrows from Input to Retinal blocks
add_arrow(ax, (7.8, 17.9), (5.2, 17.4))
add_arrow(ax, (10.2, 17.9), (12.8, 17.4))

# ========== RETINAL PREPROCESSING ==========
left_center = 4.5
right_center = 13.5

# Left Retinal
add_box(ax, left_center, 17.1, 2.2, 0.9, NAVY, [('Retinal', 11, 'normal'), ('center-surround', 9, 'italic')], textcolor='white')
ax.text(left_center, 17.7, 'Left Eye', fontsize=10, ha='center', va='center', fontweight='bold', color='#2E4053')

# Right Retinal
add_box(ax, right_center, 17.1, 2.2, 0.9, DARK_RED, [('Retinal', 11, 'normal'), ('center-surround', 9, 'italic')], textcolor='white')
ax.text(right_center, 17.7, 'Right Eye', fontsize=10, ha='center', va='center', fontweight='bold', color='#8B0000')

# ========== LEFT EYE LGN STREAM ==========
# Block 1: P, K, M
add_box(ax, left_center - 2, 15.5, 1.5, 1.2, GREEN, [('P', 14, 'normal'), ('s=1', 10, 'italic'), ('96x96', 9, 'italic')])
add_box(ax, left_center, 15.5, 1.5, 1.2, PINK, [('K', 14, 'normal'), ('s=2', 10, 'italic'), ('48x48', 9, 'italic')])
add_box(ax, left_center + 2, 15.5, 1.5, 1.2, PEACH, [('M', 14, 'normal'), ('s=3', 10, 'italic'), ('32x32', 9, 'italic')])

add_arrow(ax, (left_center - 0.5, 16.65), (left_center - 2, 16.1))
add_arrow(ax, (left_center, 16.65), (left_center, 16.1))
add_arrow(ax, (left_center + 0.5, 16.65), (left_center + 2, 16.1))

# P block 1 layer 2
add_box(ax, left_center - 2, 13.8, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (left_center - 2, 14.9), (left_center - 2, 14.25))

# K-Gate 1 (left)
add_box(ax, left_center, 13.1, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (left_center, 14.9), (left_center, 13.45))
add_arrow(ax, (left_center - 0.6, 13.1), (left_center - 1.6, 13.5), style='--')
add_arrow(ax, (left_center + 0.6, 13.1), (left_center + 1.6, 14.0), style='--')

# Block 2: P, K, M (left)
add_box(ax, left_center - 2, 11.6, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, left_center, 11.6, 1.3, 0.9, PINK, [('K', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, left_center + 2, 11.6, 1.3, 0.9, PEACH, [('M', 12, 'normal'), ('s=1', 9, 'italic')])

add_arrow(ax, (left_center - 2, 13.35), (left_center - 2, 12.05))
add_arrow(ax, (left_center, 12.75), (left_center, 12.05))
add_arrow(ax, (left_center + 2, 14.9), (left_center + 2, 12.05))

# P block 2 layer 2
add_box(ax, left_center - 2, 9.9, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (left_center - 2, 11.15), (left_center - 2, 10.35))

# K-Gate 2 (left)
add_box(ax, left_center, 9.2, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (left_center, 11.15), (left_center, 9.55))
add_arrow(ax, (left_center - 0.6, 9.2), (left_center - 1.6, 9.6), style='--')
add_arrow(ax, (left_center + 0.6, 9.2), (left_center + 1.6, 10.2), style='--')

# ========== RIGHT EYE LGN STREAM ==========
# Block 1: P, K, M
add_box(ax, right_center - 2, 15.5, 1.5, 1.2, GREEN, [('P', 14, 'normal'), ('s=1', 10, 'italic'), ('96x96', 9, 'italic')])
add_box(ax, right_center, 15.5, 1.5, 1.2, PINK, [('K', 14, 'normal'), ('s=2', 10, 'italic'), ('48x48', 9, 'italic')])
add_box(ax, right_center + 2, 15.5, 1.5, 1.2, PEACH, [('M', 14, 'normal'), ('s=3', 10, 'italic'), ('32x32', 9, 'italic')])

add_arrow(ax, (right_center - 0.5, 16.65), (right_center - 2, 16.1))
add_arrow(ax, (right_center, 16.65), (right_center, 16.1))
add_arrow(ax, (right_center + 0.5, 16.65), (right_center + 2, 16.1))

# P block 1 layer 2
add_box(ax, right_center - 2, 13.8, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (right_center - 2, 14.9), (right_center - 2, 14.25))

# K-Gate 1 (right)
add_box(ax, right_center, 13.1, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (right_center, 14.9), (right_center, 13.45))
add_arrow(ax, (right_center - 0.6, 13.1), (right_center - 1.6, 13.5), style='--')
add_arrow(ax, (right_center + 0.6, 13.1), (right_center + 1.6, 14.0), style='--')

# Block 2: P, K, M (right)
add_box(ax, right_center - 2, 11.6, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, right_center, 11.6, 1.3, 0.9, PINK, [('K', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, right_center + 2, 11.6, 1.3, 0.9, PEACH, [('M', 12, 'normal'), ('s=1', 9, 'italic')])

add_arrow(ax, (right_center - 2, 13.35), (right_center - 2, 12.05))
add_arrow(ax, (right_center, 12.75), (right_center, 12.05))
add_arrow(ax, (right_center + 2, 14.9), (right_center + 2, 12.05))

# P block 2 layer 2
add_box(ax, right_center - 2, 9.9, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (right_center - 2, 11.15), (right_center - 2, 10.35))

# K-Gate 2 (right)
add_box(ax, right_center, 9.2, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (right_center, 11.15), (right_center, 9.55))
add_arrow(ax, (right_center - 0.6, 9.2), (right_center - 1.6, 9.6), style='--')
add_arrow(ax, (right_center + 0.6, 9.2), (right_center + 1.6, 10.2), style='--')

# ========== LGN → V1 FUSION ==========
# M streams (binocular) - already at 32x32
add_box(ax, 7, 7.5, 2.0, 0.9, YELLOW, [('M fusion', 11, 'normal'), ('binocular', 9, 'italic')])

# K streams (binocular) - pooled to match M
add_box(ax, 9, 7.5, 2.0, 0.9, YELLOW, [('K fusion', 11, 'normal'), ('pool → 32x32', 9, 'italic')])

# P streams (binocular) with stride to match M
add_box(ax, 11, 7.5, 2.0, 0.9, YELLOW, [('P fusion', 11, 'normal'), ('s=3 → 32x32', 9, 'italic')])

# Arrows from LGN to fusion (M) - from M blocks at left_center+2 and right_center+2
# Left M goes straight down to M fusion
add_arrow(ax, (left_center + 2, 11.15), (6.5, 7.95), color=PEACH)
# Right M - comes from far right, use 90-degree angle to avoid crossing boxes
add_angled_arrow(ax, (right_center + 2, 11.15), (7.5, 7.95), go_horizontal_first=True, color=PEACH)
# Arrows from LGN to fusion (K) - from K blocks at left_center and right_center
# Route from bottom of K block 2 directly to K fusion
add_arrow(ax, (left_center, 8.85), (8.5, 7.95), color=PINK)
add_arrow(ax, (right_center, 8.85), (9.5, 7.95), color=PINK)
# Arrows from LGN to fusion (P) - from P blocks at left_center-2 and right_center-2
# Left P - comes from far left, use 90-degree angle to avoid crossing boxes
add_angled_arrow(ax, (left_center - 2, 9.45), (10.5, 7.95), go_horizontal_first=True, color=GREEN)
# Right P goes straight down to P fusion
add_arrow(ax, (right_center - 2, 9.45), (11.5, 7.95), color=GREEN)

# ========== V1 SIMPLE/COMPLEX PATHWAYS ==========
# V1 input concatenation
add_box(ax, 9, 5.8, 2.8, 0.8, GRAY, [('V1 Input', 11, 'normal'), ('M + P + K', 9, 'italic')])
add_arrow(ax, (7, 7.05), (8.2, 6.15))
add_arrow(ax, (9, 7.05), (9, 6.2))
add_arrow(ax, (11, 7.05), (9.8, 6.15))

# Simple pathway (stride=1, 2 layers)
add_box(ax, 6.5, 4.3, 1.8, 1.2, ORANGE, [('Simple', 12, 'normal'), ('s=1', 10, 'italic'), ('2 layers', 9, 'italic')])

# Complex pathway (stride=2, 1 layer)
add_box(ax, 11.5, 4.3, 1.8, 1.2, TEAL, [('Complex', 12, 'normal'), ('s=2', 10, 'italic'), ('1 layer', 9, 'italic')])

add_arrow(ax, (8.3, 5.4), (6.5, 4.9))
add_arrow(ax, (9.7, 5.4), (11.5, 4.9))

# ========== V1 FUSION ==========
add_box(ax, 9, 2.5, 2.5, 0.9, LAVENDER, [('V1 Fusion', 12, 'normal'), ('Simple + Complex', 9, 'italic')])

add_arrow(ax, (6.5, 3.7), (8.2, 2.85))
add_arrow(ax, (11.5, 3.7), (9.8, 2.85))

# ========== GAP + FC ==========
add_box(ax, 9, 1.0, 2.2, 0.9, MAUVE, [('GAP + FC', 14, 'normal')])
add_arrow(ax, (9, 2.05), (9, 1.45))

# Legend (bottom right)
legend_x = 16
legend_y = 3.5
ax.text(legend_x, legend_y + 2.5, 'Legend', fontsize=11, fontweight='bold', ha='center')
add_box(ax, legend_x, legend_y + 1.8, 1.2, 0.5, GREEN, [('P', 10, 'normal')])
ax.text(legend_x + 1.2, legend_y + 1.8, 'Parvo', fontsize=9, va='center')
add_box(ax, legend_x, legend_y + 1.2, 1.2, 0.5, PINK, [('K', 10, 'normal')])
ax.text(legend_x + 1.2, legend_y + 1.2, 'Konio', fontsize=9, va='center')
add_box(ax, legend_x, legend_y + 0.6, 1.2, 0.5, PEACH, [('M', 10, 'normal')])
ax.text(legend_x + 1.2, legend_y + 0.6, 'Magno', fontsize=9, va='center')
add_box(ax, legend_x, legend_y, 1.2, 0.5, ORANGE, [('S', 10, 'normal')])
ax.text(legend_x + 1.2, legend_y, 'Simple', fontsize=9, va='center')
add_box(ax, legend_x, legend_y - 0.6, 1.2, 0.5, TEAL, [('C', 10, 'normal')])
ax.text(legend_x + 1.2, legend_y - 0.6, 'Complex', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkxv1_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkxv1_architecture.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved mpkxv1_architecture.png and .pdf")
