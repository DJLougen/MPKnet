"""
Generate MPKx architecture diagram with two eye streams, matching the V4 simple style.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 16))
ax.set_xlim(0, 18)
ax.set_ylim(0, 16)
ax.axis('off')

# Colors matching the original
GRAY = '#E8E8E8'
GREEN = '#A8D5A2'  # P pathway
PINK = '#D4A5C9'   # K pathway
PEACH = '#F5CBA7'  # M pathway
BLUE = '#A9CCE3'   # K-Gate
YELLOW = '#F9E79F' # V1 Fusion
MAUVE = '#D7BDE2'  # GAP + FC

def add_box(ax, x, y, w, h, color, lines, fontsize=12, textcolor='black'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.3",
                         facecolor=color, edgecolor='#666666', linewidth=1)
    ax.add_patch(box)
    # Multi-line text
    total_lines = len(lines)
    for i, (text, fs, style) in enumerate(lines):
        offset = (total_lines - 1) / 2 - i
        ax.text(x, y + offset * 0.35, text, ha='center', va='center',
                fontsize=fs, fontstyle=style, fontweight='bold' if style == 'normal' else 'normal',
                color=textcolor)

def add_arrow(ax, start, end, style='-'):
    color = '#666666' if style == '-' else '#9B59B6'
    ls = '-' if style == '-' else '--'
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, ls=ls))

# Title
ax.text(9, 15.5, 'MPKx', fontsize=24, ha='center', va='center', fontweight='bold')
ax.text(9, 15.0, '0.214M | 3x3 | strides 1:2:3', fontsize=12, ha='center', va='center',
        fontstyle='italic', color='#666666')

# Input
add_box(ax, 9, 14.5, 2.5, 0.8, GRAY, [('Input', 12, 'normal'), ('96x96', 10, 'italic')])

# Arrows from Input to Retinal blocks
add_arrow(ax, (7.8, 14.1), (5.2, 13.6))
add_arrow(ax, (10.2, 14.1), (12.8, 13.6))

# ========== RETINAL PREPROCESSING ==========
NAVY = '#2E4053'  # Left eye color
DARK_RED = '#922B21'  # Right eye color

# Left Retinal
add_box(ax, 4.5, 13.3, 2.2, 0.9, NAVY, [('Retinal', 11, 'normal'), ('center-surround', 9, 'italic')], textcolor='white')
ax.text(4.5, 13.9, 'Left Eye', fontsize=10, ha='center', va='center', fontweight='bold', color='#2E4053')

# Right Retinal
add_box(ax, 13.5, 13.3, 2.2, 0.9, DARK_RED, [('Retinal', 11, 'normal'), ('center-surround', 9, 'italic')], textcolor='white')
ax.text(13.5, 13.9, 'Right Eye', fontsize=10, ha='center', va='center', fontweight='bold', color='#8B0000')

# ========== LEFT EYE STREAM (centered at x=4.5) ==========
left_center = 4.5

# Block 1: P, K, M
add_box(ax, left_center - 2, 12, 1.5, 1.2, GREEN, [('P', 14, 'normal'), ('s=1', 10, 'italic'), ('96x96', 9, 'italic')])
add_box(ax, left_center, 12, 1.5, 1.2, PINK, [('K', 14, 'normal'), ('s=2', 10, 'italic'), ('48x48', 9, 'italic')])
add_box(ax, left_center + 2, 12, 1.5, 1.2, PEACH, [('M', 14, 'normal'), ('s=3', 10, 'italic'), ('32x32', 9, 'italic')])

# Arrows from Retinal to pathways
add_arrow(ax, (left_center - 0.5, 12.9), (left_center - 2, 12.6))
add_arrow(ax, (left_center, 12.9), (left_center, 12.6))
add_arrow(ax, (left_center + 0.5, 12.9), (left_center + 2, 12.6))

# P block 1 layer 2
add_box(ax, left_center - 2, 10.2, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (left_center - 2, 11.4), (left_center - 2, 10.65))

# K-Gate 1 (left)
add_box(ax, left_center, 9.5, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (left_center, 11.4), (left_center, 9.85))
# Dashed arrows from K-Gate to P and M
add_arrow(ax, (left_center - 0.6, 9.5), (left_center - 1.6, 10.0), style='--')
add_arrow(ax, (left_center + 0.6, 9.5), (left_center + 1.6, 10.5), style='--')

# Block 2: P, K, M (left)
add_box(ax, left_center - 2, 8.0, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, left_center, 8.0, 1.3, 0.9, PINK, [('K', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, left_center + 2, 8.0, 1.3, 0.9, PEACH, [('M', 12, 'normal'), ('s=1', 9, 'italic')])

add_arrow(ax, (left_center - 2, 9.75), (left_center - 2, 8.45))
add_arrow(ax, (left_center, 9.15), (left_center, 8.45))
add_arrow(ax, (left_center + 2, 11.4), (left_center + 2, 8.45))

# P block 2 layer 2
add_box(ax, left_center - 2, 6.2, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (left_center - 2, 7.55), (left_center - 2, 6.65))

# K-Gate 2 (left)
add_box(ax, left_center, 5.5, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (left_center, 7.55), (left_center, 5.85))
# Dashed arrows
add_arrow(ax, (left_center - 0.6, 5.5), (left_center - 1.6, 5.9), style='--')
add_arrow(ax, (left_center + 0.6, 5.5), (left_center + 1.6, 6.5), style='--')

# ========== RIGHT EYE STREAM (centered at x=13.5) ==========
right_center = 13.5

# Block 1: P, K, M
add_box(ax, right_center - 2, 12, 1.5, 1.2, GREEN, [('P', 14, 'normal'), ('s=1', 10, 'italic'), ('96x96', 9, 'italic')])
add_box(ax, right_center, 12, 1.5, 1.2, PINK, [('K', 14, 'normal'), ('s=2', 10, 'italic'), ('48x48', 9, 'italic')])
add_box(ax, right_center + 2, 12, 1.5, 1.2, PEACH, [('M', 14, 'normal'), ('s=3', 10, 'italic'), ('32x32', 9, 'italic')])

# Arrows from Retinal to pathways
add_arrow(ax, (right_center - 0.5, 12.9), (right_center - 2, 12.6))
add_arrow(ax, (right_center, 12.9), (right_center, 12.6))
add_arrow(ax, (right_center + 0.5, 12.9), (right_center + 2, 12.6))

# P block 1 layer 2
add_box(ax, right_center - 2, 10.2, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (right_center - 2, 11.4), (right_center - 2, 10.65))

# K-Gate 1 (right)
add_box(ax, right_center, 9.5, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (right_center, 11.4), (right_center, 9.85))
# Dashed arrows
add_arrow(ax, (right_center - 0.6, 9.5), (right_center - 1.6, 10.0), style='--')
add_arrow(ax, (right_center + 0.6, 9.5), (right_center + 1.6, 10.5), style='--')

# Block 2: P, K, M (right)
add_box(ax, right_center - 2, 8.0, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, right_center, 8.0, 1.3, 0.9, PINK, [('K', 12, 'normal'), ('s=1', 9, 'italic')])
add_box(ax, right_center + 2, 8.0, 1.3, 0.9, PEACH, [('M', 12, 'normal'), ('s=1', 9, 'italic')])

add_arrow(ax, (right_center - 2, 9.75), (right_center - 2, 8.45))
add_arrow(ax, (right_center, 9.15), (right_center, 8.45))
add_arrow(ax, (right_center + 2, 11.4), (right_center + 2, 8.45))

# P block 2 layer 2
add_box(ax, right_center - 2, 6.2, 1.3, 0.9, GREEN, [('P', 12, 'normal'), ('3x3', 9, 'italic')])
add_arrow(ax, (right_center - 2, 7.55), (right_center - 2, 6.65))

# K-Gate 2 (right)
add_box(ax, right_center, 5.5, 1.5, 0.7, BLUE, [('K-Gate', 11, 'normal')])
add_arrow(ax, (right_center, 7.55), (right_center, 5.85))
# Dashed arrows
add_arrow(ax, (right_center - 0.6, 5.5), (right_center - 1.6, 5.9), style='--')
add_arrow(ax, (right_center + 0.6, 5.5), (right_center + 1.6, 6.5), style='--')

# ========== V1 FUSION ==========
add_box(ax, 9, 3.5, 3.0, 0.9, YELLOW, [('V1 Fusion', 14, 'normal')])

# Arrows to V1 from both streams (P and M only, not K)
add_arrow(ax, (left_center - 2, 5.75), (7.8, 3.8))
add_arrow(ax, (left_center + 2, 7.55), (8.2, 3.95))
add_arrow(ax, (right_center - 2, 5.75), (9.8, 3.8))
add_arrow(ax, (right_center + 2, 7.55), (10.2, 3.95))

# ========== GAP + FC ==========
add_box(ax, 9, 1.8, 2.2, 0.9, MAUVE, [('GAP + FC', 14, 'normal')])
add_arrow(ax, (9, 3.05), (9, 2.25))

plt.tight_layout()
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_architecture.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_architecture.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved mpkx_architecture.png and .pdf")
