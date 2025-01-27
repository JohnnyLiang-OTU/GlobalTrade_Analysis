import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pickle 

def main():
#--Data/Variables--
    # Get the data parameters from the user
    Year_input = input("Which year [1995, 2005, 2015, 2020] of the dataset would you like to graph?: ")
    trade_agreement_name = input("Enter a trade agreement name ('EEA', 'USMCA', 'RCEP', 'AfCFTA', 'MERCOSUR'): ").strip()
    
    # Load the necessary data
    trade_agreements_df = pd.read_csv('data/trade_agreement.csv')
    dataset = pd.read_csv(f'data/renamed_{Year_input}.csv')
    
    
#--World Top 10 Graph--
    # Create the global graph (world-wide trade network)
    G_world = Graph_Builder(Year_input)
    
    # Calculate the trade volume for the world
    # Types: "Total", "Export", or "Import".
    trade_volume_world = calculate_trade_volume(G_world, volume_type="Total")
    
    # Get the top 10 economies by total trade volume
    number_of_countries = 10
    top_X_economies = sorted(trade_volume_world.items(), key=lambda x: x[1], reverse=True)[:number_of_countries]
    top_X_countries = [economy[0] for economy in top_X_economies]
    
    # Create the subgraph for the top 10 economies (global trade network)
    G_top_X = G_world.subgraph(top_X_countries).copy()
    
    # Print the graph for the top 10 economies (global graph)
    print_graph(G_top_X, trade_volume_world, "Top 10 Economies (Global Trade Network)")
    
    
#--Trade Agreement Graph--
    # Create the subgraph for the given trade agreement
    G_agreement = trade_agreement_subgraph(trade_agreement_name, dataset, trade_agreements_df)
    
    # Calculate trade volume for the trade agreement graph
    trade_volume_agreement = calculate_trade_volume(G_agreement, volume_type="Total")
    
    # Filter for top 15 economies in the trade agreement
    top_15_economies = sorted(trade_volume_agreement.items(), key=lambda x: x[1], reverse=True)[:15]
    top_15_countries = [economy[0] for economy in top_15_economies]
    G_agreement_top_15 = G_agreement.subgraph(top_15_countries).copy()
    
    # Print the graph for the trade agreement subgraph (top 15 economies only)
    print_graph(G_agreement_top_15, trade_volume_agreement, f"Top 15 in {trade_agreement_name} Agreement")


    
#--Main Functions--
def print_graph(G, trade_volume, graph_title):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use circular layout for more evenly spaced nodes
    pos = nx.circular_layout(G)

    # Separate edges based on trade direction
    export_edges = [(u, v, G[u][v]['weight']) for u, v in G.edges if G[u][v]['weight'] > 0]


    # Edge weights for coloring (positive for exports, negative for imports)
    export_weights = [weight for _, _, weight in export_edges]

    # Color mapping for edges
    cmap_edges = plt.get_cmap('Oranges')  # Use the 'Oranges' colormap for imports

    # Step 5: Adjust the node sizes based on trade volume
    # Normalize the trade volume
    trade_volumes = [trade_volume[country] for country in G.nodes()]
    max_trade_volume = max(trade_volumes)
    min_trade_volume = min(trade_volumes)

    # Normalize the trade volume to a range between 0 and 1
    normalized_sizes = [(volume - min_trade_volume) / (max_trade_volume - min_trade_volume) for volume in trade_volumes]

    # Scale the normalized sizes to make the nodes larger
    node_sizes = [size * 5000 + 1000 for size in normalized_sizes]  # The "+1000" ensures a minimum size
    
    #Used for arrowhead spacing from nodes    
    node_radii = {node: np.sqrt(size / np.pi) for node, size in zip(G.nodes, node_sizes)}

    # Draw edges for exports
    if export_edges:
        export_margins = [
            (node_radii[u], node_radii[v]) for u, v, _ in export_edges
        ]
        for (u, v, weight), (source_margin, target_margin) in zip(export_edges, export_margins):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], width=2, edge_color=[weight],
                edge_cmap=cmap_edges, edge_vmin=min(export_weights), edge_vmax=max(export_weights),
                ax=ax, connectionstyle="arc3,rad=0.25", arrowstyle='-|>', arrowsize=12,
                min_source_margin=source_margin -2 , min_target_margin=target_margin - 2)

    # Draw nodes with the adjusted sizes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='black', ax=ax)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax, font_color='white', font_weight='bold', bbox=dict(facecolor='grey', alpha=0.8))
    
    # Set figure background color to black
    fig.patch.set_facecolor('grey')

    # Title
    ax.set_title(f"Trade Network of {graph_title}")

    # Hide axis
    ax.axis('off')

    # Add color bar for export and import trade values
    if export_weights:
        sm_export = plt.cm.ScalarMappable(cmap=cmap_edges)
        sm_export.set_array([])  # Empty array to avoid warnings
        fig.colorbar(sm_export, ax=ax, label="Export Trade Value")
        
        
    # Show plot
    plt.show()
    
def Graph_Builder(year_input, ) -> nx.Graph:
    G = nx.DiGraph()
    df = pd.read_csv(f'data/renamed_{year_input}.csv')

    for index, row in df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        # CSV data is by thousands.
        # In the csv, a value of 1 equals true value of 1,000.
        value = row['Value'] * 1000 
        
        # Adds edge from exporting country to importing country.
        G.add_edge(exporter, importer, weight=value)
        
    # Save the graph to a file
    nx.write_graphml(G, os.path.join(".", f"trade_network_{year_input}.graphml"))
    
    return G

def trade_agreement_subgraph(trade_agreement_name, dataset, trade_agreements_df):
    # Get countries in the specified trade agreement
    countries_in_agreement = trade_agreements_df[trade_agreements_df['Trade Agreement'] == trade_agreement_name]['Country']
    
    # Filter the dataset for only those countries
    filtered_df = dataset[dataset['Exporter'].isin(countries_in_agreement) & dataset['Importer'].isin(countries_in_agreement)]
    
    # Create the subgraph based on the filtered data
    G = nx.DiGraph()
    for index, row in filtered_df.iterrows():
        exporter = row['Exporter']
        importer = row['Importer']
        value = row['Value'] * 1000  # Adjust value by 1000 as per your dataset scale
        G.add_edge(exporter, importer, weight=value)
    
    return G

def calculate_trade_volume(G, volume_type):
    trade_volume = {}
    
    for node in G.nodes:
        if volume_type == "Total":
            # Sum of both exports (outgoing) and imports (incoming)
            exports = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            imports = sum(G[neighbor][node]['weight'] for neighbor in G.predecessors(node))
            trade_volume[node] = exports + imports
        
        elif volume_type == "Export":
            # Sum of exports (outgoing trade)
            exports = sum(G[node][neighbor]['weight'] for neighbor in G.neighbors(node))
            trade_volume[node] = exports
        
        elif volume_type == "Import":
            # Sum of imports (incoming trade)
            imports = sum(G[neighbor][node]['weight'] for neighbor in G.predecessors(node))
            trade_volume[node] = imports

    return trade_volume



def print_exports_imports(G, top_economies):
    # Display the total exports and imports for each of the top 10 economies
    for country in top_economies:
        # Calculate total exports (outgoing trade)
        exports = sum(G[country][neighbor]['weight'] for neighbor in G.neighbors(country))
        
        # Calculate total imports (incoming trade)
        imports = sum(G[neighbor][country]['weight'] for neighbor in G.predecessors(country))
        
        # Format the output as currency
        print(f"{country}: Exports = ${exports:,.2f}, Imports = ${imports:,.2f}")
        
def get_all_countries_with_X(dictionary):
    all_countries_with_X = set()
    for _, val in dictionary.items():
        all_countries_with_X.update(val)
    return all_countries_with_X

if __name__ == "__main__":
    main()