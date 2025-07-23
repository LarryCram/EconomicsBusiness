import duckdb

db = duckdb.connect('/home/lc/m/econ.duckdb')
db.sql("SHOW ALL TABLES").show()


def edge_weighted_pagerank(links, damping_factor=0.85, iterations=100):
    """
    Calculates edge-weighted PageRank using dictionaries.

    Args:
        links (dict): A dictionary where keys are source nodes and values are
                      dictionaries mapping target nodes to edge weights.
                      Example: {'A': {'B': 0.5, 'C': 0.3}, 'B': {'A': 0.2}}
        damping_factor (float): The probability of following a link (d).
        iterations (int): The number of iterations to run the algorithm.

    Returns:
        dict: A dictionary mapping nodes to their calculated PageRank scores.
    """
    all_nodes = set()
    for source, targets in links.items():
        all_nodes.add(source)
        for target in targets:
            all_nodes.add(target)

    num_nodes = len(all_nodes)
    pagerank_scores = {node: 1.0 / num_nodes for node in all_nodes}

    for _ in range(iterations):
        new_pagerank_scores = {node: 0.0 for node in all_nodes}
        
        for source_node, target_links in links.items():
            total_outgoing_weight = sum(target_links.values())
            if total_outgoing_weight == 0:  # Handle dangling nodes
                # Distribute PageRank evenly among all nodes if no outgoing links
                for node in all_nodes:
                    new_pagerank_scores[node] += pagerank_scores[source_node] / num_nodes
            else:
                for target_node, weight in target_links.items():
                    contribution = (pagerank_scores[source_node] * weight) / total_outgoing_weight
                    new_pagerank_scores[target_node] += contribution

        # Apply damping factor and random jump
        for node in all_nodes:
            new_pagerank_scores[node] = (damping_factor * new_pagerank_scores[node]) + \
                                       ((1 - damping_factor) / num_nodes)

        pagerank_scores = new_pagerank_scores

    # Normalize scores to ensure they sum to 1
    total_pagerank = sum(pagerank_scores.values())
    normalized_pagerank = {node: score / total_pagerank for node, score in pagerank_scores.items()}

    return normalized_pagerank

# Example Usage:
example_links = {
    'A': {'B': 1.0, 'C': 0.5},
    'B': {'C': 1.0},
    'C': {'A': 0.8}
}

pagerank_results = edge_weighted_pagerank(example_links)
print(pagerank_results)