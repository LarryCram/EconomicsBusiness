from configuration.make_toy_model import make_model
from rankers.single_unit_ranks import unit_citation_matrix, unit_driver
from rankers.multi_unit_ranks import multi_unit_driver
from utils.algorithms import verify_pn_geller
from configuration.network_visualization import plot_citation_network, analyze_network_properties

if __name__ == "__main__":
    # Generate toy model data
    data = make_model(verbose=True)
    
    # Run single-unit citation analysis  
    single_results = unit_driver(data, verbose=True)

    # Run multi-unit citation analysis with default weights (1/3, 1/3, 1/3)
    print("\n" + "="*80)
    print("MULTI-UNIT CITATION ANALYSIS RESULTS")
    print("="*80)
    
    multi_results = multi_unit_driver(data, verbose=True)
    
    # Demonstrate custom weighting
    print("\n" + "="*80)
    print("MULTI-UNIT CITATION ANALYSIS WITH CUSTOM WEIGHTS")
    print("="*80)
    
    # # Example: Give more weight to journal citations, less to author/institution
    # custom_weights = {'journal': 0.6, 'author': 0.2, 'institution': 0.2}
    # custom_weights = {'journal': 0.5, 'author': 0.0, 'institution': 0.5}
    # multi_results_weighted = multi_unit_driver(data, weights=custom_weights, verbose=True)
    
    # Network visualization after multi-unit analysis
    print("\n" + "="*80)
    print("CITATION NETWORK VISUALIZATION")
    print("="*80)
    
    # Set up matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')
    
    print("Analyzing network properties...")
    network_properties = analyze_network_properties(data, verbose=True)
    
    print("\nCreating citation network visualization...")
    plot_citation_network(data, save_path="main_citation_network.png", figsize=(16, 10), 
                         ranking_results=multi_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Generated files:")
    print("  - ../plots/main_citation_network.png: Citation network visualization")
    print("  - Console output: Ranking analysis results")
