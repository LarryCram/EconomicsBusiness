import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
try:
    from .make_toy_model import make_model
except ImportError:
    from make_toy_model import make_model


def calculate_predicted_geller_from_pn(pn_scores, citation_matrix):
    """
    Calculate predicted Geller scores from PN scores using the relationship:
    Geller[i] ∝ PN[i] / output[i] where output[i] = total citations made by unit i
    
    Args:
        pn_scores (dict): Dictionary of PN scores keyed by unit ID
        citation_matrix (pd.DataFrame): Citation matrix with units as rows/columns
        
    Returns:
        dict: Predicted Geller scores normalized to sum to 1
    """
    import numpy as np
    
    predicted_geller = {}
    
    # Get unique unit types from the PN scores
    unit_types = set()
    for unit_id in pn_scores.keys():
        if '_' in unit_id:
            unit_type = unit_id.split('_')[0]
            unit_types.add(unit_type)
    
    # Calculate outputs for each unit (row sums from citation matrix)
    outputs = citation_matrix.sum(axis=1) if hasattr(citation_matrix, 'sum') else {}
    
    # Calculate ratios PN[i] / output[i] for each unit
    ratios = []
    unit_order = []
    
    for unit_id in pn_scores.keys():
        pn_score = pn_scores[unit_id]
        
        # Find corresponding output (total citations made by this unit)
        output = outputs.get(unit_id, 0.0) if outputs else 0.0
        
        if output > 0:
            ratio = pn_score / output
        else:
            ratio = 0.0
        
        ratios.append(ratio)
        unit_order.append(unit_id)
    
    # Normalize ratios to get predicted Geller scores
    ratios = np.array(ratios)
    if ratios.sum() > 0:
        normalized_ratios = ratios / ratios.sum()
    else:
        normalized_ratios = ratios
    
    # Create predicted Geller scores dictionary
    for i, unit_id in enumerate(unit_order):
        predicted_geller[unit_id] = normalized_ratios[i]
    
    return predicted_geller


def plot_citation_network(model_df=None, save_path=None, figsize=(12, 8), show_labels=True, ranking_results=None):
    """
    Create a matplotlib figure of the citation network using NetworkX.
    
    Args:
        model_df (pd.DataFrame, optional): Unnested DataFrame from make_model(). 
                                         If None, will call make_model() to get data.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
        figsize (tuple): Figure size (width, height) in inches.
        show_labels (bool): Whether to show node labels.
        ranking_results (dict, optional): Results from multi_unit_driver containing influence scores.
        
    Returns:
        tuple: (fig, ax, G) - matplotlib figure, axes, and NetworkX graph
    """
    
    # Get data if not provided
    if model_df is None:
        model_df = make_model(verbose=False)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Get comprehensive work metadata including all authors and institutions
    works_info = []
    for work_id in model_df['work_id'].unique():
        work_data = model_df[model_df['work_id'] == work_id]
        
        # Get basic info
        journal = work_data['journal_id'].iloc[0]
        year = work_data['publication_year'].iloc[0]
        referenced_works = work_data['referenced_works'].iloc[0]
        
        # Get all authors and institutions for this work
        authors = sorted(work_data['author_id'].unique())
        institutions = sorted(work_data['institution_id'].unique())
        
        # Create comprehensive label with optional influence scores
        authors_str = ','.join(authors)
        institutions_str = ','.join(institutions)
        
        # Base label
        label = f"{work_id}\nJ:{journal} A:{authors_str}\nI:{institutions_str} Y:{year}"
        
        # Add influence scores if provided
        if ranking_results and 'rankings' in ranking_results:
            score_lines = []
            predicted_geller_scores = None
            
            # Add Pinski-Narin scores if available
            if 'pinski_narin' in ranking_results['rankings']:
                pn_scores = ranking_results['rankings']['pinski_narin']  #['pi']
                
                # Get scores for this work's entities
                journal_score = pn_scores.get(f'J_{journal}', 0.0)
                author_scores = [pn_scores.get(f'A_{author}', 0.0) for author in authors]
                institution_scores = [pn_scores.get(f'I_{inst}', 0.0) for inst in institutions]
                
                # Calculate averages
                avg_author_score = sum(author_scores) / len(author_scores) if author_scores else 0.0
                avg_inst_score = sum(institution_scores) / len(institution_scores) if institution_scores else 0.0
                
                score_lines.append(f"PN: J={journal_score:.3f} A={avg_author_score:.3f} I={avg_inst_score:.3f}")
                
                # Calculate predicted Geller scores from PN scores if citation matrix is available
                if 'citation_matrix' in ranking_results:
                    predicted_geller_scores = calculate_predicted_geller_from_pn(
                        pn_scores, ranking_results['citation_matrix']
                    )
            
            # Add Geller scores if available
            if 'geller' in ranking_results['rankings']:
                geller_scores = ranking_results['rankings']['geller']  #['pi']
                
                # Get scores for this work's entities
                journal_score = geller_scores.get(f'J_{journal}', 0.0)
                author_scores = [geller_scores.get(f'A_{author}', 0.0) for author in authors]
                institution_scores = [geller_scores.get(f'I_{inst}', 0.0) for inst in institutions]
                
                # Calculate averages
                avg_author_score = sum(author_scores) / len(author_scores) if author_scores else 0.0
                avg_inst_score = sum(institution_scores) / len(institution_scores) if institution_scores else 0.0
                
                score_lines.append(f"GL: J={journal_score:.3f} A={avg_author_score:.3f} I={avg_inst_score:.3f}")
            
            # Add predicted Geller scores if available
            if predicted_geller_scores:
                # Get predicted scores for this work's entities
                journal_pred_score = predicted_geller_scores.get(f'J_{journal}', 0.0)
                author_pred_scores = [predicted_geller_scores.get(f'A_{author}', 0.0) for author in authors]
                institution_pred_scores = [predicted_geller_scores.get(f'I_{inst}', 0.0) for inst in institutions]
                
                # Calculate averages
                avg_author_pred_score = sum(author_pred_scores) / len(author_pred_scores) if author_pred_scores else 0.0
                avg_inst_pred_score = sum(institution_pred_scores) / len(institution_pred_scores) if institution_pred_scores else 0.0
                
                score_lines.append(f"PG: J={journal_pred_score:.3f} A={avg_author_pred_score:.3f} I={avg_inst_pred_score:.3f}")
            
            # Add score lines to label
            for score_line in score_lines:
                label += f"\n{score_line}"
        
        works_info.append({
            'work_id': work_id,
            'journal_id': journal,
            'publication_year': year,
            'referenced_works': referenced_works,
            'authors': authors,
            'institutions': institutions,
            'label': label
        })
    
    # Add nodes for each work with comprehensive labels
    for work in works_info:
        work_id = work['work_id']
        
        # Add node with all metadata
        G.add_node(work_id, 
                  journal=work['journal_id'], 
                  year=work['publication_year'],
                  authors=work['authors'],
                  institutions=work['institutions'],
                  label=work['label'])
    
    # Add edges for citations
    for work in works_info:
        citing_work = work['work_id']
        referenced_works = work['referenced_works']
        
        if isinstance(referenced_works, list) and referenced_works:
            for cited_work in referenced_works:
                # Add edge from citing work to cited work
                G.add_edge(citing_work, cited_work)
    
    # Create the plot with adjusted positioning for left labels
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Adjust subplot position to leave more space on the left for labels
    # [left, bottom, width, height] in figure coordinates (0-1)
    ax.set_position([0.25, 0.1, 0.7, 0.8])  # Move plot right and shrink slightly
    
    # Set up layout - use hierarchical layout based on publication year
    pos = {}
    
    # Group nodes by year for hierarchical layout
    year_groups = {}
    for node in G.nodes():
        year = G.nodes[node]['year']
        if year not in year_groups:
            year_groups[year] = []
        year_groups[year].append(node)
    
    # Position nodes in layers by year
    y_positions = {}
    years = sorted(year_groups.keys())
    for i, year in enumerate(years):
        y_positions[year] = len(years) - i - 1  # Newer works higher
    
    # Calculate positions
    for year, nodes in year_groups.items():
        y = y_positions[year]
        num_nodes = len(nodes)
        
        if num_nodes == 1:
            x_positions = [0]
        else:
            x_positions = [(i - (num_nodes-1)/2) * 2 for i in range(num_nodes)]
        
        # Position nodes normally (no shift)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (x_positions[i], y)
    
    # Color nodes by journal
    journals = list(set(G.nodes[node]['journal'] for node in G.nodes()))
    journal_colors = plt.cm.Set3(range(len(journals)))
    journal_color_map = dict(zip(journals, journal_colors))
    
    node_colors = [journal_color_map[G.nodes[node]['journal']] for node in G.nodes()]
    
    # Draw nodes with larger size to accommodate more text
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=4000,  # Increased size for more text
                          alpha=0.7,
                          ax=ax)
    
    # Draw labels if requested - offset to the left of nodes
    if show_labels:
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        # Create offset positions for labels (move right by 0.6 * horizontal spacing = 1.2)
        label_pos = {}
        for node, (x, y) in pos.items():
            label_pos[node] = (x - 1.5 + 1.2, y)  # Original offset -1.5, then move right by 1.2
        
        nx.draw_networkx_labels(G, label_pos, labels, 
                               font_size=10,  # Increased from 7 to 10 (50% increase)
                               font_weight='bold',
                               horizontalalignment='right',  # Changed from center to right
                               ax=ax)
    
    # Draw edges with larger arrows OVER the nodes
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=40,  # Doubled arrow size
                          arrowstyle='->',
                          alpha=1.0,  # Fully opaque edges
                          width=2,  # Thicker edges to make arrows more visible
                          ax=ax)
    
    # Customize the plot
    ax.set_title('Citation Network\n(Arrows point from citing work to cited work)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Create legend for journals
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=journal_color_map[journal], 
                                 markersize=10, label=journal)
                      for journal in sorted(journals)]
    ax.legend(handles=legend_elements, title='Journals', 
             loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add year labels on the far left
    for year, y_pos in y_positions.items():
        ax.text(-max([abs(x) for x, y in pos.values()]) - 3.0, y_pos,  # Moved further left
               f'Year {year}', 
               fontsize=10, fontweight='bold', 
               verticalalignment='center')
    
    # Remove axis
    ax.set_axis_off()
    
    # Remove axis ticks and make background transparent
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Set axis limits to ensure labels are visible
    if pos:
        all_x = [x for x, y in pos.values()]
        all_y = [y for x, y in pos.values()]
        # Extend left boundary to accommodate labels
        x_min = min(all_x) - 2.5  # Extra space for labels
        x_max = max(all_x) + 1.0
        y_min = min(all_y) - 0.5
        y_max = max(all_y) + 0.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save or show
    if save_path:
        # Save to plots folder if no directory specified
        if '/' not in save_path:
            save_path = f"../plots/{save_path}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, ax, G


def plot_network_statistics(model_df=None, save_path=None, figsize=(15, 10)):
    """
    Create a comprehensive plot showing network statistics and multiple views.
    
    Args:
        model_df (pd.DataFrame, optional): Unnested DataFrame from make_model()
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size (width, height) in inches
        
    Returns:
        tuple: (fig, axes) - matplotlib figure and axes array
    """
    
    # Get data if not provided
    if model_df is None:
        model_df = make_model(verbose=False)
    
    # Create the main network graph
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Main citation network
    ax1 = axes[0, 0]
    _, _, G = plot_citation_network(model_df, show_labels=True, figsize=(6, 6))
    
    # Recreate the network plot in the subplot with comprehensive labels
    works_info = []
    for work_id in model_df['work_id'].unique():
        work_data = model_df[model_df['work_id'] == work_id]
        
        journal = work_data['journal_id'].iloc[0]
        year = work_data['publication_year'].iloc[0]
        referenced_works = work_data['referenced_works'].iloc[0]
        authors = sorted(work_data['author_id'].unique())
        institutions = sorted(work_data['institution_id'].unique())
        
        works_info.append({
            'work_id': work_id,
            'journal_id': journal,
            'publication_year': year,
            'referenced_works': referenced_works,
            'authors': authors,
            'institutions': institutions
        })
    
    G = nx.DiGraph()
    
    for work in works_info:
        work_id = work['work_id']
        G.add_node(work_id, 
                  journal=work['journal_id'], 
                  year=work['publication_year'],
                  authors=work['authors'],
                  institutions=work['institutions'])
    
    for work in works_info:
        citing_work = work['work_id']
        referenced_works = work['referenced_works']
        if isinstance(referenced_works, list) and referenced_works:
            for cited_work in referenced_works:
                G.add_edge(citing_work, cited_work)
    
    # Use spring layout for this view
    pos = nx.spring_layout(G, seed=42)
    
    journals = list(set(G.nodes[node]['journal'] for node in G.nodes()))
    journal_colors = plt.cm.Set3(range(len(journals)))
    journal_color_map = dict(zip(journals, journal_colors))
    node_colors = [journal_color_map[G.nodes[node]['journal']] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                          node_size=1000, alpha=0.7)
    
    # Draw labels offset to the left
    labels = {node: node for node in G.nodes()}
    label_pos = {}
    for node, (x, y) in pos.items():
        label_pos[node] = (x - 0.15, y)  # Smaller offset for spring layout
    
    nx.draw_networkx_labels(G, label_pos, labels, ax=ax1,
                           font_size=12, horizontalalignment='center')  # Increased from 8 to 12 (50% increase)
    
    # Draw edges over nodes
    nx.draw_networkx_edges(G, pos, ax=ax1,
                          arrows=True, arrowsize=30, arrowstyle='->',
                          edge_color='gray', alpha=1.0, width=1.5)  # Fully opaque edges
    
    ax1.set_title('Citation Network (Spring Layout)', fontweight='bold')
    
    # Plot 2: Degree distribution
    ax2 = axes[0, 1]
    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    
    x = range(len(G.nodes()))
    width = 0.35
    ax2.bar([i - width/2 for i in x], in_degrees, width, label='In-degree (cited)', alpha=0.7)
    ax2.bar([i + width/2 for i in x], out_degrees, width, label='Out-degree (citing)', alpha=0.7)
    ax2.set_xlabel('Works')
    ax2.set_ylabel('Degree')
    ax2.set_title('Citation Degrees by Work', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(G.nodes()))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Timeline view
    ax3 = axes[1, 0]
    years = [G.nodes[node]['year'] for node in G.nodes()]
    year_counts = pd.Series(years).value_counts().sort_index()
    
    ax3.bar(year_counts.index, year_counts.values, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Publication Year')
    ax3.set_ylabel('Number of Works')
    ax3.set_title('Publications by Year', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Journal distribution
    ax4 = axes[1, 1]
    journal_counts = pd.Series([G.nodes[node]['journal'] for node in G.nodes()]).value_counts()
    
    colors = [journal_color_map[journal] for journal in journal_counts.index]
    ax4.pie(journal_counts.values, labels=journal_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax4.set_title('Distribution by Journal', fontweight='bold')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        # Save to plots folder if no directory specified
        if '/' not in save_path:
            save_path = f"../plots/{save_path}"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network statistics plot saved to: {save_path}")
    else:
        plt.show()
    
    return fig, axes


def analyze_network_properties(model_df=None, verbose=True):
    """
    Analyze and print network properties.
    
    Args:
        model_df (pd.DataFrame, optional): Unnested DataFrame from make_model()
        verbose (bool): Whether to print detailed analysis
        
    Returns:
        dict: Dictionary containing network properties
    """
    
    # Get data if not provided
    if model_df is None:
        model_df = make_model(verbose=False)
    
    # Build the graph with comprehensive metadata
    works_info = []
    for work_id in model_df['work_id'].unique():
        work_data = model_df[model_df['work_id'] == work_id]
        
        journal = work_data['journal_id'].iloc[0]
        year = work_data['publication_year'].iloc[0]
        referenced_works = work_data['referenced_works'].iloc[0]
        authors = sorted(work_data['author_id'].unique())
        institutions = sorted(work_data['institution_id'].unique())
        
        works_info.append({
            'work_id': work_id,
            'journal_id': journal,
            'publication_year': year,
            'referenced_works': referenced_works,
            'authors': authors,
            'institutions': institutions
        })
    
    G = nx.DiGraph()
    
    for work in works_info:
        work_id = work['work_id']
        G.add_node(work_id, 
                  journal=work['journal_id'], 
                  year=work['publication_year'],
                  authors=work['authors'],
                  institutions=work['institutions'])
    
    for work in works_info:
        citing_work = work['work_id']
        referenced_works = work['referenced_works']
        if isinstance(referenced_works, list) and referenced_works:
            for cited_work in referenced_works:
                G.add_edge(citing_work, cited_work)
    
    # Calculate properties
    properties = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_strongly_connected': nx.is_strongly_connected(G),
        'is_weakly_connected': nx.is_weakly_connected(G),
        'num_strongly_connected_components': nx.number_strongly_connected_components(G),
        'num_weakly_connected_components': nx.number_weakly_connected_components(G),
    }
    
    # Add degree statistics
    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]
    
    properties.update({
        'avg_in_degree': sum(in_degrees) / len(in_degrees) if in_degrees else 0,
        'avg_out_degree': sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        'max_in_degree': max(in_degrees) if in_degrees else 0,
        'max_out_degree': max(out_degrees) if out_degrees else 0,
    })
    
    if verbose:
        print("NETWORK ANALYSIS")
        print("=" * 40)
        print(f"Nodes (works): {properties['num_nodes']}")
        print(f"Edges (citations): {properties['num_edges']}")
        print(f"Network density: {properties['density']:.3f}")
        print(f"Strongly connected: {properties['is_strongly_connected']}")
        print(f"Weakly connected: {properties['is_weakly_connected']}")
        print(f"Average in-degree: {properties['avg_in_degree']:.2f}")
        print(f"Average out-degree: {properties['avg_out_degree']:.2f}")
        print(f"Max in-degree: {properties['max_in_degree']}")
        print(f"Max out-degree: {properties['max_out_degree']}")
        
        # Show most cited and most citing works
        in_degree_ranking = sorted([(node, G.in_degree(node)) for node in G.nodes()], 
                                  key=lambda x: x[1], reverse=True)
        out_degree_ranking = sorted([(node, G.out_degree(node)) for node in G.nodes()], 
                                   key=lambda x: x[1], reverse=True)
        
        print(f"\nMost cited works:")
        for work, degree in in_degree_ranking[:3]:
            journal = G.nodes[work]['journal']
            year = G.nodes[work]['year']
            authors = ','.join(G.nodes[work]['authors'])
            institutions = ','.join(G.nodes[work]['institutions'])
            print(f"  {work} (J:{journal}, A:{authors}, I:{institutions}, Y:{year}): {degree} citations")
        
        print(f"\nMost citing works:")
        for work, degree in out_degree_ranking[:3]:
            journal = G.nodes[work]['journal']
            year = G.nodes[work]['year']
            authors = ','.join(G.nodes[work]['authors'])
            institutions = ','.join(G.nodes[work]['institutions'])
            print(f"  {work} (J:{journal}, A:{authors}, I:{institutions}, Y:{year}): {degree} references")
    
    return properties


if __name__ == "__main__":
    # Example usage
    print("Creating citation network visualization...")
    
    # Generate the data
    data = make_model(verbose=True)
    
    # Analyze network properties
    properties = analyze_network_properties(data)
    
    # Create visualizations
    print("\nCreating network plots...")
    
    # Simple network plot
    plot_citation_network(data, save_path="citation_network.png")
    
    # Comprehensive statistics plot
    plot_network_statistics(data, save_path="network_statistics.png")