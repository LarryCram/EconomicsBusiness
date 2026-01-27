#!/usr/bin/env python3

"""
Simple network visualization script that can be run from the main toy_models folder.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'configuration'))

from configuration.make_toy_model import make_model
from configuration.network_visualization import plot_citation_network, analyze_network_properties

def create_network_plot(save_name="network_plot.png"):
    """Create a citation network plot and save it."""
    
    # Set up matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')
    
    print("Creating citation network visualization...")
    
    # Generate data
    data = make_model(verbose=False)
    
    # Analyze the network
    print("\nNetwork Analysis:")
    properties = analyze_network_properties(data, verbose=True)
    
    # Create the plot
    print(f"\nCreating plot: {save_name}")
    plot_citation_network(data, save_path=save_name, figsize=(12, 8))
    
    print(f"✓ Network plot saved as: {save_name}")
    return data, properties

if __name__ == "__main__":
    create_network_plot()