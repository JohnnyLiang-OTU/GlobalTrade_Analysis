import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graphing_csv import Graph_Builder, print_graph, calculate_trade_volume

def main():
    # Decoder - CSV FILE - cols: country_code,country_name,continent
    # --------------------------------------------------------------
    decoder = pd.read_csv("data/decoder_with_continents_dict.csv")

    
    # Main Dataset - CSV FILE - cols: Exporter,Importer,Value
    # -------------------------------
    dataset = pd.read_csv("data/cleaned_2020.csv")
    


    # Gather all countries by continent.
    # Results in a dictionary where key = continent ; value = set containing all countries belonging to continent
    continents_dictionary = {}
    for _, v in decoder.iterrows():
        if pd.notna(v["continent"]):
            if v["continent"] not in continents_dictionary:
                continents_dictionary[v["continent"]] = {v["country_code"]}
            else: 
                continents_dictionary.get(v["continent"]).add(v["country_code"])



    # Gathers all countries by trade_agreement.
    # Uses trade_agreement.csv ; cols: Country Code,Country,Trade Agreement
    # Results in a dictionary where key = trade agreement ; value = set containing all countries belonging to the trade agreement.
    #------------------
    agreements_dictionary = {}

    trade_agreements_df = pd.read_csv("data/trade_agreement.csv")
    for _, v in trade_agreements_df.iterrows(): 
        if v["Trade Agreement"] not in agreements_dictionary:
            agreements_dictionary[v["Trade Agreement"]] = {v["Country Code"]}
        else: 
            agreements_dictionary.get(v["Trade Agreement"]).add(v["Country Code"])
    
    # ---------------- Code may start here -----------------

    G = Graph_Builder_by_Continent(dataset, "Europe", decoder)
    
    # G = Graph_Builder("2020")
    #G = Graph_Builder_2("2020")
    communities = nx.algorithms.community.louvain_communities(G, "weight")
    for community in communities:
       print(community)
    

    # modularity_TA_A(dataset, agreements_dictionary)
    # modularity_TA_B(dataset, agreements_dictionary)
    # louvain_within_continent(dataset, decoder)
    modularity_Continent(dataset, continents_dictionary)
    
    # G = Graph_Builder_by_Continent(dataset, "Asia", decoder)
    # pos = nx.circular_layout(G)
    # nx.draw(G)
    # plt.show()
    # edge_distribution_plot(G)
    # degree_distribution_plot_in_out(G)
    degree_distribution_plot_total(G)
    plt.show()
    #avg_deg = (sum(dict(G.degree()).values()) / len(G.nodes))/2
    #print(f"\nThe average degree of the graph is: {avg_deg:.2f}")
    
    #print(calc_diameter(G))
    #print(calc_transitivity(G))

    
    # top_10, bottom_10 = get_top_bottom_degrees(G, top_n=5)
    # print("\nTop 10 nodes with highest degrees:")
    # for node, degree in top_10:
    #     print(f"Node {node} has degree {degree/2}")

    # print("\nBottom 10 nodes with lowest degrees:")
    # for node, degree in bottom_10:
    #     print(f"Node {node} has degree {degree/2}")
    
    # top_in, top_out = top_in_out_degrees(G, top_n=5)
    # print("\nTop 10 nodes with highest in-degrees:")
    # for node, degree in top_in:
    #     print(f"Node {node} has in-degree {degree/2}")
        
    # print("\nTop 10 nodes with highest out-degrees:")
    # for node, degree in top_out:
    #     print(f"Node {node} has out-degree {degree/2}")

    
    # top_clustering_nodes = top_weighted_clustering_coefficients(G, top_n=10)
    # print("\nTop nodes with highest weighted local clustering coefficients:")
    # for node, coeff in top_clustering_nodes:
    #     print(f"Node {node} has a weighted clustering coefficient of {coeff:.4f}")
        
        
    #avg_distance = average_distance(G)
    #print(f"The average distance between nodes is: {avg_distance:.4f}")
    
    
    # top_eigenvector_nodes = top_eigenvector_centrality(G, top_n=10, weight='weight')
    # print("Top nodes with highest eigenvector centrality:")
    # for node, centrality in top_eigenvector_nodes:
    #     print(f"Node {node}: Eigenvector Centrality = {centrality:.4f}")
       
        
    # top_hubs, top_authorities = top_hits_scores(G, top_n=10)
    # print("Top Hubs:")
    # for node, score in top_hubs:
    #     print(f"Node {node}: Hub Score = {score:.4f}")

    # print("\nTop Authorities:")
    # for node, score in top_authorities:
    #     print(f"Node {node}: Authority Score = {score:.4f}")
    
    
    # avg_centralities = average_centralities_with_inverted_weight(G, weight='weight')
    # print("Average Centralities:")
    # for centrality, avg_value in avg_centralities.items():
    #     print(f"{centrality}: {avg_value:.4f}" if avg_value is not None else f"{centrality}: Could not calculate")
       
        
    # top_centralities_result = top_centralities_with_inverted_weight(G, top_n=5, weight='weight')
    # print("Top Centralities:")
    # for centrality, nodes in top_centralities_result.items():
    #     if nodes is not None:
    #         print(f"\n{centrality}:")
    #         for node, value in nodes:
    #             print(f"Node {node}: {value:.4f}")
    #     else:
    #         print(f"\n{centrality}: Could not calculate")
    
    #print(density_by_TA(G, agreements_dictionary))
    
    #print(total_density(G))
                    

"""
Builds a world-wide graph.
Then Modularity is Calculated based on a partition defined by trade agreements.
Countries that are not part of a trade agreement are added to a no trade agreement set. Important fact.


For Example:
Nodes China and North Korea are both part of the RCEP trade agreement. They belong to the RCEP community.
Nodes Mexico and Canada are both part of USMCA trade agreement. They belong to the same USMCA community.
RCEP and USMCA are 2 communities within a partition that contains other communities
"""
def modularity_TA_A(dataset, agreements_dictionary):
    G = nx.DiGraph()
    
    for index, row in dataset.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        # CSV data is by thousands.
        # In the csv, a value of 1 equals true value of 1,000.
        value = row['Value'] * 1000 
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)

    node_list = [int(x) for x in G.nodes]
    one_set_agreements = get_all_countries_with_X(agreements_dictionary)
    no_agreement_list = get_no_agreements(node_list, one_set_agreements)
    
    partition_list = [set(x) for x in agreements_dictionary.values()]
    partition_list.append(no_agreement_list)

    modularity = nx.algorithms.community.modularity(G, partition_list)
    print(f"Modularity of a Worldwide Graph partitioning by trade agreements: {modularity}")

"""
Builds a graph and calculates modularity by partitioning by Trade Agreements.
Only edges where both exporters and importers belong to ANY trade agreement will be included.
Then modularity is calculated using a partition based on trade agreements. [{TA1}, {TA2}, {TA3}, ...]

Different to TA_A, this method builds a graph that only includes nodes and edges where both are part of an agreement.
"""
def modularity_TA_B(dataset, agreements_dictionary):
    countries_in_agreements = get_all_countries_with_X(agreements_dictionary)
    filtered_df = dataset[dataset['Importer'].isin(countries_in_agreements) & dataset['Exporter'].isin(countries_in_agreements)]

    G = nx.DiGraph()
    
    for index, row in filtered_df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        # CSV data is by thousands.
        # In the csv, a value of 1 equals true value of 1,000.
        value = row['Value'] * 1000 
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)

    print("\n")
    partition_list = [set(x) for x in agreements_dictionary.values()]
    modularity = nx.algorithms.community.modularity(G, partition_list)
    print(f"Modularity of a trade-agreement exclusive graph partioned by trade agreements: {modularity}\n")
    return partition_list

"""
Builds a graph that partitions based on continents.
Only the nodes and edges that go from a country with a continent to another country with a continent will show up.
*Important* not all exporters and importers belong to a continent. There's some places not categorized as a country and their continent is ambiguous.

Then calculates modularity using a partition with each community being a different continent.

"""
def modularity_Continent(dataset, continents_dictionary):
    one_set_continental_countries = get_all_countries_with_X(continents_dictionary)
    filtered_df = dataset[dataset["Exporter"].isin(one_set_continental_countries) &
                          dataset["Importer"].isin(one_set_continental_countries)]

    G = nx.DiGraph()    
    for index, row in filtered_df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        value = row['Value']
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)
    
        
    partition_list = [set(x) & set(G.nodes) for x in continents_dictionary.values()]
    
    modularity = nx.algorithms.community.modularity(G, partition_list)
    print(f"Modularity of a graph partitioned by Continents: {modularity}\n")
    return partition_list

"""
Builds a Graph based on countries of ONE continent.

Then finds the best partition within this continent through Louvain.
And calculates the modularity of set partition.
"""
def louvain_within_continent(dataset, decoder):
    target_continent = "Africa"
    
    # Gets all countries codes that belong to target_continent
    countries_in_target_continent = decoder[decoder['continent'] == target_continent]['country_code']
    
    # Dictionary that helps with translating a country code to a country name.
    transformer = dict(zip(decoder['country_code'], decoder['country_name']))
    
    
    filtered_df = dataset[
            dataset['Exporter'].isin(countries_in_target_continent) &
            dataset['Importer'].isin(countries_in_target_continent)
        ]

    G = nx.DiGraph()

    for index, row in filtered_df.iterrows():
    # for index, row in dataset.iterrows():
            exporter = get_country_name(row['Exporter'], transformer)
            importer = get_country_name(row['Importer'], transformer)
            # CSV data is by thousands.
            # In the csv, a value of 1 equals true value of 1,000.
            value = row['Value'] * 1000 
            
            # Adds edge from exporting country to importing country.
            G.add_edge(exporter, importer, weight=value)
        
    communities = nx.algorithms.community.louvain_communities(G)
    print(communities)
    modularity = nx.algorithms.community.modularity(G, communities)
    print(modularity)

def get_top_bottom_degrees(graph, top_n=10):
    """
    Finds the nodes with the highest and lowest degrees in the graph.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    top_n (int): Number of nodes to retrieve for top/bottom degrees.

    Returns:
    tuple: Two lists - top `top_n` nodes and bottom `top_n` nodes with their degrees.
    """
    degree_dict = dict(graph.degree())  # Get degrees as a dictionary
    sorted_by_degree = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)

    # Get top and bottom nodes
    top_nodes = sorted_by_degree[:top_n]
    bottom_nodes = sorted_by_degree[-top_n:]

    return top_nodes, bottom_nodes

def top_in_out_degrees(graph, top_n=10):
    """
    Find the top nodes with the highest in-degrees and out-degrees in a directed graph.

    Parameters:
    graph (networkx.DiGraph): The directed graph to analyze.
    top_n (int): Number of top nodes to return for in/out degrees.

    Returns:
    tuple: Two lists containing the top nodes with their in-degrees and out-degrees.
    """
    # Check if the graph is directed
    if not graph.is_directed():
        raise ValueError("The graph must be directed.")

    # In-degrees and out-degrees
    in_degree_dict = dict(graph.in_degree())
    out_degree_dict = dict(graph.out_degree())

    # Sort by in-degree and out-degree
    top_in_degrees = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_out_degrees = sorted(out_degree_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_in_degrees, top_out_degrees

def top_weighted_clustering_coefficients(graph, top_n=10):
    """
    Finds the nodes with the highest weighted local clustering coefficients in the graph.

    Parameters:
    graph (networkx.Graph): The graph to analyze (should include weights on edges).
    top_n (int): Number of top nodes to retrieve.

    Returns:
    list: Top `top_n` nodes with their weighted local clustering coefficients.
    """
    # Calculate local weighted clustering coefficients
    clustering_coeffs = nx.clustering(graph, weight='weight')
    
    # Sort nodes by clustering coefficient in descending order
    sorted_clustering = sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top `top_n` nodes
    top_nodes = sorted_clustering[:top_n]
    
    return top_nodes

def calc_diameter(graph):
    
    if nx.is_strongly_connected(graph):
        # If the graph is strongly connected, calculate its diameter
        diameter = nx.diameter(graph)
        print(f"The diameter of the strongly connected directed graph is: {diameter}")
    else:
        # Get the largest strongly connected component (SCC)
        largest_scc = max(nx.strongly_connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_scc)
        diameter = nx.diameter(subgraph)
        print(f"The graph is not strongly connected. Diameter of the largest SCC is: {diameter}")
        
def calc_transitivity(graph):
    
    transitivity = nx.transitivity(graph)
    print(f"The transitivity (global clustering coefficient) of the directed graph is: {transitivity:.4f}")
    
def average_distance(graph):
    """
    Computes the average shortest path length (distance) between nodes in the graph.

    Parameters:
    graph (networkx.Graph): The graph to analyze. Should be connected.

    Returns:
    float: The average shortest path length between all pairs of nodes.
    """
    # Ensure the graph is connected
    if not nx.is_strongly_connected(graph):
        raise ValueError("The graph must be connected to compute average distance.")

    # Compute the average shortest path length
    avg_distance = nx.average_shortest_path_length(graph)
    
    return avg_distance

def top_eigenvector_centrality(graph, top_n=10, weight=None, tol=1e-6, max_iter=100):
    """
    Finds the nodes with the highest eigenvector centrality in the graph.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    top_n (int): Number of top nodes to retrieve.
    weight (str or None): The edge attribute to use as weight. Default is None (unweighted).
    tol (float): Tolerance for convergence in eigenvector calculation.
    max_iter (int): Maximum number of iterations for eigenvector computation.

    Returns:
    list: Top `top_n` nodes with their eigenvector centrality scores.
    """
    # Compute eigenvector centrality
    centrality = nx.eigenvector_centrality(graph, weight=weight, max_iter=max_iter, tol=tol)

    # Sort nodes by eigenvector centrality in descending order
    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)

    # Get the top `top_n` nodes
    top_nodes = sorted_centrality[:top_n]

    return top_nodes

def top_hits_scores(graph, top_n=10, max_iter=100, tol=1e-8, normalized=True):
    """
    Calculates HITS scores (hubs and authorities) for a directed graph and returns the top nodes.

    Parameters:
    graph (networkx.DiGraph): The directed graph to analyze.
    top_n (int): Number of top nodes to retrieve for hub and authority scores.
    max_iter (int): Maximum number of iterations for convergence.
    tol (float): Tolerance for convergence.
    normalized (bool): Whether to normalize the scores.

    Returns:
    tuple: Two lists containing the top `top_n` nodes by hub scores and authority scores.
    """
    # Compute HITS scores
    hits_scores = nx.hits(graph, max_iter=max_iter, tol=tol, normalized=normalized)
    hub_scores = hits_scores[0]  # Hubs
    authority_scores = hits_scores[1]  # Authorities

    # Sort nodes by hub and authority scores
    top_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_authorities = sorted(authority_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return top_hubs, top_authorities

def average_centralities_with_inverted_weight(graph, weight='weight', tol=1e-6, max_iter=100):
    """
    Calculates the average centralities for betweenness, closeness, degree, and eigenvector centralities,
    using inverted weights where higher weight indicates closeness.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    weight (str or None): The edge attribute to use as weight. Default is 'weight'.
    tol (float): Tolerance for convergence in eigenvector calculation.
    max_iter (int): Maximum number of iterations for eigenvector computation.

    Returns:
    dict: A dictionary with the average values of each centrality measure.
    """
    # Invert weights by creating a copy of the graph with inverted weights
    inverted_graph = graph.copy()
    if weight:
        for u, v, data in inverted_graph.edges(data=True):
            if weight in data and data[weight] != 0:
                data[weight] = 1 / data[weight]  # Invert weight (closer = higher weight)

    # Betweenness Centrality (using inverted weights)
    betweenness = nx.betweenness_centrality(inverted_graph, weight=weight)
    avg_betweenness = sum(betweenness.values()) / len(betweenness) if betweenness else 0

    # Closeness Centrality (using inverted weights)
    closeness = nx.closeness_centrality(inverted_graph, distance=weight)
    avg_closeness = sum(closeness.values()) / len(closeness) if closeness else 0

    # Degree Centrality (remains unaffected by weight inversion)
    degree = nx.degree_centrality(graph)
    avg_degree = sum(degree.values()) / len(degree) if degree else 0

    # Eigenvector Centrality (uses original weights)
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight=weight, max_iter=max_iter, tol=tol)
        avg_eigenvector = sum(eigenvector.values()) / len(eigenvector) if eigenvector else 0
    except nx.PowerIterationFailedConvergence:
        avg_eigenvector = None
        print("Warning: Eigenvector centrality calculation did not converge.")

    return {
        "Average Betweenness Centrality": avg_betweenness,
        "Average Closeness Centrality": avg_closeness,
        "Average Degree Centrality": avg_degree,
        "Average Eigenvector Centrality": avg_eigenvector,
    }



def top_centralities_with_inverted_weight(graph, top_n=5, weight='weight', tol=1e-6, max_iter=100):
    """
    Finds the top N nodes with the highest betweenness, closeness, degree, and eigenvector centralities,
    using inverted weights where higher weight indicates closeness.

    Parameters:
    graph (networkx.Graph): The graph to analyze.
    top_n (int): Number of top nodes to retrieve for each centrality.
    weight (str or None): The edge attribute to use as weight. Default is 'weight'.
    tol (float): Tolerance for convergence in eigenvector calculation.
    max_iter (int): Maximum number of iterations for eigenvector computation.

    Returns:
    dict: A dictionary with top nodes for each centrality.
    """
    # Invert weights by creating a copy of the graph with inverted weights
    inverted_graph = graph.copy()
    if weight:
        for u, v, data in inverted_graph.edges(data=True):
            if weight in data and data[weight] != 0:
                data[weight] = 1 / data[weight]  # Invert weight (closer = higher weight)

    # Betweenness Centrality (using inverted weights)
    betweenness = nx.betweenness_centrality(inverted_graph, weight=weight)
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Closeness Centrality (using inverted weights)
    closeness = nx.closeness_centrality(inverted_graph, distance=weight)
    top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Degree Centrality (remains unaffected by weight inversion)
    degree = nx.degree_centrality(graph)
    top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Eigenvector Centrality (using original weights)
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight=weight, max_iter=max_iter, tol=tol)
        top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:top_n]
    except nx.PowerIterationFailedConvergence:
        top_eigenvector = None
        print("Warning: Eigenvector centrality calculation did not converge.")

    return {
        "Top Betweenness Centrality": top_betweenness,
        "Top Closeness Centrality": top_closeness,
        "Top Degree Centrality": top_degree,
        "Top Eigenvector Centrality": top_eigenvector,
    }

def density_by_TA(graph, agreements_dictionary):
    density_by_agreement = {}
    
    print(graph.nodes())

    for agreement, countries in agreements_dictionary.items():
        # Create a subgraph for the current trade agreement
        subgraph = graph.subgraph(countries)
        
        # Calculate the density of the subgraph
        density = nx.density(subgraph)
        density_by_agreement[agreement] = density

    # Print results
    for agreement, density in density_by_agreement.items():
        print(f"Trade Agreement '{agreement}': Density = {density:.4f}")

    return density_by_agreement

def Graph_Builder_2(year_input, ) -> nx.Graph:
    G = nx.DiGraph()
    df = pd.read_csv(f'data/cleaned_{year_input}.csv')

    for index, row in df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        # CSV data is by thousands.
        # In the csv, a value of 1 equals true value of 1,000.
        value = row['Value'] * 1000 
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)
        
    
    return G

def total_density(graph):
    
    density = nx.density(graph)
    print(f"The global density of the world graph is: {density}")
    

# Helper Functions 
# ----------------

# Translates country code to country name
def get_country_name(country_code, transformer):
        return transformer.get(country_code, "Unknown Country")


# Merges all sets of countries with X into one large set.
# X : by continent || by trade agreement
# In other words, a continent dictionary or a trade dictionary are both valid parameters for this function.
def get_all_countries_with_X(dictionary):
    all_countries_with_X = set()
    for _, val in dictionary.items():
        all_countries_with_X.update(val)
    return all_countries_with_X

# Gets all the countries with no trade agreements into one set.
def get_no_agreements(available_country_codes, totality):
    no_agreement_list = set()
    for element in available_country_codes:
         if element not in totality:
              no_agreement_list.add(element)
    return no_agreement_list


def edge_distribution_plot(G: nx.Graph) -> None:
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    plt.hist(edge_weights, bins=10, edgecolor='black', alpha=0.75)
    plt.xscale('log')
    plt.title("Logarithmic Edge Weight Distribution")
    plt.xlabel("Edge Weight (log scale)")
    plt.ylabel("Frequency")
    
def degree_distribution_plot_total(G: nx.DiGraph) -> None:
    degrees = [deg for _, deg in G.degree()]
    bins = np.logspace(np.log10(min(degrees)), np.log10(max(degrees)), 25)
    plt.hist(degrees, bins=25, edgecolor='black')
    # plt.xscale('log')
    # plt.xlim(0, max(degrees) + 10)
    plt.title("Total Degree Distribution")
    plt.xlabel("Total Degrees")
    plt.ylabel("Frequency")
    plt.tight_layout()

def degree_distribution_plot_in_out(G: nx.DiGraph) -> None:
    # Calculate in-degrees and out-degrees
    in_degrees = [deg for _, deg in G.in_degree()]
    out_degrees = [deg for _, deg in G.out_degree()]

    # Create subplots for in-degree and out-degree
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # In-degree histogram
    # in_bins = np.logspace(np.log10(min(in_degrees)), np.log10(max(in_degrees)), 25)
    axes[0].hist(in_degrees, bins=25, edgecolor='black', alpha=0.75)
    # axes[0].set_xscale('log')
    axes[0].set_xlim(0, max(in_degrees) + 10)
    axes[0].set_title("In-Degree Distribution")
    axes[0].set_xlabel("In-Degree")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Out-degree histogram
    # out_bins = np.logspace(np.log10(min(out_degrees)), np.log10(max(out_degrees)), 25)
    axes[1].hist(out_degrees, bins=25, edgecolor='black', alpha=0.75, color='orange')
    # axes[1].set_xscale('log')
    axes[1].set_xlim(0, max(out_degrees) + 10)
    axes[1].set_title("Out-Degree Distribution")
    axes[1].set_xlabel("Out-Degree")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

# Builds a Graph given a continent.
def Graph_Builder_by_Continent(dataset, target_continent, decoder):
    countries_in_target_continent = decoder[decoder['continent'] == target_continent]['country_code']
    transformer = dict(zip(decoder['country_code'], decoder['country_name']))

    filtered_df = dataset[
            dataset['Exporter'].isin(countries_in_target_continent) &
            dataset['Importer'].isin(countries_in_target_continent)
        ]
    
    G = nx.DiGraph()

    for index, row in filtered_df.iterrows():
    # for index, row in dataset.iterrows():
            exporter = get_country_name(row['Exporter'], transformer)
            importer = get_country_name(row['Importer'], transformer)
            value = row['Value']
            
            # Adds edge from exporting country to importing country.
            G.add_edge(exporter, importer, weight=value)
        
    return G

def Graph_Builder_Inv(year_input, ) -> nx.Graph:
    G = nx.DiGraph()
    df = pd.read_csv(f'data/renamed_{year_input}.csv')

    for index, row in df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        value = 1 / row['Value']
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)
        
    # Save the graph to a file    
    return G

    
if __name__ == "__main__":
    main()