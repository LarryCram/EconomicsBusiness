# Citation Network Visualization

This folder contains functions to create matplotlib visualizations of the citation network defined by the toy model data using NetworkX.

## Files

- `network_visualization.py`: Main visualization functions
- `network_demo.py`: Comprehensive demonstration script  
- `../create_network_plot.py`: Simple script to create a basic network plot

## Key Functions

### `plot_citation_network(model_df, save_path, figsize, show_labels)`
Creates a citation network visualization with:
- Nodes representing works (papers)
- Directed edges representing citations
- Color coding by journal
- Hierarchical layout by publication year
- Node labels showing work ID, journal, and year

### `plot_network_statistics(model_df, save_path, figsize)`
Creates a comprehensive 4-panel visualization showing:
1. Citation network (spring layout)
2. Degree distribution (in-degree vs out-degree)
3. Publications by year timeline
4. Journal distribution pie chart

### `analyze_network_properties(model_df, verbose)`
Analyzes and reports network properties:
- Node/edge counts
- Network density
- Connectivity analysis
- Degree statistics
- Most cited/citing works

## Usage Examples

### Simple Usage
```python
from configuration.make_toy_model import make_model
from configuration.network_visualization import plot_citation_network

# Generate data and create plot
data = make_model()
plot_citation_network(data, save_path="my_network.png")
```

### Comprehensive Analysis
```python
from configuration.network_visualization import *

# Get data
data = make_model()

# Analyze properties
properties = analyze_network_properties(data, verbose=True)

# Create multiple visualizations
plot_citation_network(data, save_path="citation_network.png")
plot_network_statistics(data, save_path="network_stats.png")
```

### From Main Directory
```python
# Run from toy_models/ directory
python create_network_plot.py
```

## Network Structure

The toy model creates a citation network with:
- **8 works** (W1-W8) published across 3 years
- **3 journals** (J1, J2, J3) with color coding
- **4 authors** (A1-A4) and **3 institutions** (I1-I3)
- **13 citation relationships** forming a directed graph

### Key Network Properties
- **Density**: 23.2% (relatively sparse)
- **Connectivity**: Weakly connected (not strongly connected)
- **Most cited**: W1 (4 citations) - early influential work
- **Most citing**: W6 (3 references) - comprehensive later work

## Visualization Features

### Citation Network Plot
- **Layout**: Hierarchical by publication year (newer works on top)
- **Node colors**: Different color for each journal
- **Node labels**: Work ID, journal, and publication year
- **Arrows**: Point from citing work to cited work
- **Legend**: Shows journal color mapping

### Network Statistics Plot
- **Spring layout**: Alternative network view
- **Degree distribution**: Bar chart showing citation patterns
- **Timeline**: Publications by year
- **Journal distribution**: Pie chart of journal representation

## Requirements

- NetworkX: `pip install networkx`
- Matplotlib: `pip install matplotlib`
- Pandas: Already included in project

## Output Files

Running the visualization functions creates PNG files:
- `citation_network.png`: Main hierarchical network view
- `network_statistics.png`: Comprehensive 4-panel analysis
- `network_plot.png`: Simple network plot (from create_network_plot.py)

All plots are saved at 300 DPI for high quality output suitable for publications or presentations.