#!/usr/bin/env python3

"""
Example script showing how to use the network visualization functions.
"""

from make_toy_model import make_model
from network_visualization import (
    plot_citation_network, 
    plot_network_statistics, 
    analyze_network_properties
)

def main():
    """Demonstrate network visualization capabilities."""
    
    print("Citation Network Visualization Demo")
    print("=" * 40)
    
    # Generate the toy model data
    print("1. Generating toy model data...")
    data = make_model(verbose=True)
    
    # Analyze network properties
    print("\n2. Analyzing network properties...")
    properties = analyze_network_properties(data, verbose=True)
    
    # Create basic citation network plot
    print("\n3. Creating citation network visualization...")
    fig1, ax1, graph = plot_citation_network(
        data, 
        save_path="citation_network.png",
        figsize=(12, 8),
        show_labels=True
    )
    
    # Create comprehensive statistics plot
    print("\n4. Creating network statistics visualization...")
    fig2, axes = plot_network_statistics(
        data,
        save_path="network_statistics.png", 
        figsize=(15, 10)
    )
    
    print("\n5. Visualization files created:")
    print("   - citation_network.png: Main citation network")
    print("   - network_statistics.png: Comprehensive network analysis")
    
    # Print some interesting insights
    print(f"\n6. Key Network Insights:")
    print(f"   - Total works: {properties['num_nodes']}")
    print(f"   - Total citations: {properties['num_edges']}")
    print(f"   - Network density: {properties['density']:.1%}")
    print(f"   - Average citations per work: {properties['avg_in_degree']:.1f}")
    print(f"   - Most cited work has {properties['max_in_degree']} citations")
    print(f"   - Network is {'strongly' if properties['is_strongly_connected'] else 'weakly'} connected")
    
    return data, properties, (fig1, fig2)

if __name__ == "__main__":
    # Set up matplotlib for non-interactive use if needed
    import matplotlib
    try:
        matplotlib.use('Agg')  # Use non-interactive backend
    except:
        pass
    
    main()