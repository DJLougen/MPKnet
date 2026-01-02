"""
Generate architecture graph for MPKx using torchviz.
"""
import sys
sys.path.insert(0, '/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/private')

import torch
from torchviz import make_dot
from mpknet_v4 import BinocularMPKNetV4

# Create model
model = BinocularMPKNetV4(num_classes=200, ch=48, use_stereo=True)
model.eval()

# Create dummy input
x = torch.randn(1, 3, 64, 64)

# Forward pass
y = model(x)

# Generate graph
dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
dot.attr(rankdir='TB')  # Top to bottom layout
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')

# Save as PNG and PDF
dot.render('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_architecture_graph', format='png', cleanup=True)
dot.render('/Users/djl/Documents/uncleFesterFuntTime/mpknet_release/figures/mpkx_architecture_graph', format='pdf', cleanup=True)

print("Saved mpkx_architecture_graph.png and mpkx_architecture_graph.pdf")
