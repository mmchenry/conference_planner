import pandas as pd
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import community as community_louvain
from networkx.algorithms.community import girvan_newman
# from networkx.algorithms.community import spectral_clustering  
from networkx.algorithms.community import label_propagation_communities
from sklearn.cluster import SpectralClustering
import itertools




def calc_similarity(data_root, presentation_type='talks'):
    """
    Finds the cosine similarity between each pair of presentations and writes the results to a csv file.
    Parameters:
    - data_root (str): Path to the data directory.
    - presentation_type (str): Type of presentations to analyze. Options are 'talks', 'posters', or 'both'.
    Returns:
    - None
    """

    # Types of presentations
    if presentation_type=='talks':
        presentations = ['talks']
    elif presentation_type=='posters':     
        presentations = ['posters']
    elif presentation_type=='both':
        presentations = ['talks','posters'] 
    else:
        raise ValueError(f"Invalid presentation type '{presentation_type}'.")

    # Load keyword weights
    df_weights = pd.read_excel(os.path.join(data_root, 'keyword_weights.xlsx'))

    # Loop thru talk and poster files
    for pres_type in presentations:

        # Input and output files
        in_path    = os.path.join(data_root, 'contributed', pres_type + '_ratings.csv')
        out_path   = os.path.join(data_root, 'contributed', pres_type + '_similarity.csv')

        # Load abstract data for presentation type
        df = pd.read_csv(in_path)

        # Check for duplicate IDs
        duplicate_ids = df['id'].duplicated()
        if duplicate_ids.any():
            raise ValueError(f"Duplicate IDs found in {pres_type} data: {duplicate_ids}")

        # Set the 'id' column as the index of the DataFrame
        df = df.set_index('id')

        # Number of presentations
        num_pres = df.shape[0]

        # Get weighted ratings from df and df_weights
        df_ratings = extract_ratings(df, df_weights)

        # Calculate cosine similarity
        similarity = cosine_similarity(df_ratings)

        # Convert to a DataFrame for better readability
        similarity_df = pd.DataFrame(similarity)
        similarity_df.index = df.index
        similarity_df.columns = df.index

        # Replace diagonal and lower triangle with NaNs
        np.fill_diagonal(similarity_df.values, np.nan)
        similarity_df.values[np.tril_indices_from(similarity_df.values, -1)] = np.nan

        # Calculate the 99% quantile of the non-NaN values
        # quantile_99 = similarity_df.stack().quantile(0.99)
        # similarity_df = similarity_df.applymap(lambda x: x / quantile_99 if pd.notnull(x) else np.nan)

        # Normalize the DataFrame by the maximum value
        max_value = similarity_df.stack().max()
        similarity_df = similarity_df.applymap(lambda x: x / max_value if pd.notnull(x) else np.nan)


        # Write to disk
        similarity_df.to_csv(out_path, index=True)
        print(f"Saved {pres_type} similarity data to {out_path}")


def ana_network(data_root, presentation_type='talks', echo_graph=False, community_detection='greedy', sim_threshold=0.8):
    """
    Creates graph from similarity values, finds communities in the graph using the specified method, and plots the results.
    Parameters:
    - data_root (str): Path to the data directory.
    - presentation_type (str): Type of presentations to analyze. Options are 'talks', 'posters', or 'both'.
    - echo_graph (bool): Whether to print information about the graph.
    - community_detection (str): Community detection method. Options are 'greedy', 'louvain', 'girvan_newman', 'spectral', or 'label_propagation'.
    - sim_threshold (float): Minimum similarity between abstracts to be considered connected.
    Returns:
    - None
    """

    if (presentation_type == 'talks') or (presentation_type == 'posters'):
        presentations = [presentation_type]
    elif presentation_type == 'both':
        presentations = ['talks', 'posters']
    else:
        raise ValueError(f"Invalid presentation type '{presentation_type}'.")

    # Loop thru talk and poster files
    for pres_type in presentations:
        
        # Load abstract data for presentation type and similarity data
        full_path = os.path.join(data_root, 'contributed', pres_type + '.csv')
        sim_path = os.path.join(data_root, 'contributed', pres_type + '_similarity.csv')
        df_sim = pd.read_csv(sim_path, index_col='id')
        df = pd.read_csv(full_path)

        # Make sure abstract ids are integers
        df_sim.columns = df_sim.columns.astype(int)
        df_sim.index   = df_sim.index.astype(int)

        # Create an empty graph
        G = nx.Graph()

        # Add nodes
        for abstract_id in df_sim.columns:
            node = int(abstract_id)
            G.add_node(node)
            if not isinstance(node, int):
                raise TypeError(f"Node {node} is not an integer.")

        # Check for consistent node types
        node_types = set(type(node) for node in G.nodes())
        if len(node_types) > 1:
            raise TypeError("Inconsistent node types found in the graph.")

        # Add edges based on similarity
        for i, row in df_sim.iterrows():
            for j, similarity in row.items():
                if j > i and float(similarity) > sim_threshold:
                    G.add_edge(i, j, weight=float(similarity))

        # Check for consistent edge weight types
        weight_types = set(type(data['weight']) for _, _, data in G.edges(data=True))
        if len(weight_types) > 1:
            raise TypeError("Inconsistent edge weight types found in the graph.")

        if echo_graph:
            # Print basic information about the graph
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}")

            # Print a few nodes
            print("Some example nodes:", list(G.nodes)[:5])

            # Print a few edges with weights
            print("Some example edges with weights:")
            for u, v, data in list(G.edges(data=True))[:5]:
                print(f"({u}, {v}) - Weight: {data.get('weight', 'N/A')}")
            
        # Find communities in the graph
        communities = find_communities(G, community_detection=community_detection)

        # Print the communities
        if echo_graph:
            for community in communities:
                print(list(community))

        # Add community letters to the DataFrame df, using the 'id' column as the index
        df['community'] = ''
        for idx, community in enumerate(communities):
            for node in community:
                df.loc[df['id'] == node, 'community'] = chr(idx + 65)

        # save the df to csv
        df.to_csv(os.path.join(data_root, 'contributed', pres_type + '.csv'), index=False)
        print(f"Saved {pres_type} data to {os.path.join(data_root, 'contributed', pres_type + '.csv')}")

        # # Optionally, visualize the graph
        # pos = nx.spring_layout(G)
        # nx.draw_networkx_nodes(G, pos, node_size=700)
        # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        # nx.draw_networkx_labels(G, pos, font_size=12)
        # plt.show()
            
        plotly_network_graph(G, communities)


def find_communities(G, community_detection='greedy', num_communities=5):
    """
    Finds communities in the graph using the specified method.
    Parameters:
    - G (nx.Graph): Graph object.
    - community_detection (str): Community detection method. Options are 'greedy', 'louvain', 'girvan_newman', 'spectral', or 'label_propagation'.
    - num_communities (int): Number of communities to split the graph into (only used for 'girvan_newman' and 'spectral' methods).
    Returns:
    - communities (list): List of communities.
    """

     # Community detection
    if community_detection == 'greedy':
        communities = nx.algorithms.community.greedy_modularity_communities(G)

    # Louvain Method: The Louvain method is popular for its efficiency and effectiveness in detecting communities in large networks. It's not included in NetworkX by default, but you can use the python-louvain package, which integrates well with NetworkX.
    elif community_detection == 'louvain':
        partition = community_louvain.best_partition(G)
        communities = []
        for com in set(partition.values()):
            list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
            communities.append(list_nodes)

    # Girvan-Newman Algorithm: This algorithm detects communities by progressively removing edges from the original graph. You can specify the number of communities you want to split the graph into.
    elif community_detection == 'girvan_newman':
        k = num_communities  # Desired number of communities
        comp = girvan_newman(G)
        communities = list(itertools.takewhile(lambda c: len(c) <= k, comp))

        if communities:
            final_communities = communities[-1]
        else:
            print("No communities found or fewer communities than expected")
            final_communities = []       

    # Spectral Clustering:Spectral clustering uses eigenvalues of a matrix to reduce dimensionality before clustering in fewer dimensions. It can be particularly effective for graphs with strong community structure.           
    elif community_detection == 'spectral':
        k = num_communities  # Desired number of communities
        sc = SpectralClustering(n_clusters=k, affinity='precomputed')
        try:
            labels = sc.fit_predict(nx.to_numpy_array(G))
        except ValueError as e:
            print("Error in spectral clustering:", e)
            print("The graph might not be fully connected.")
            # Handle the error, e.g., by skipping community detection or using a different method
            labels = []

        # Map each node to its community label
        node_to_community = {list(G.nodes())[i]: label for i, label in enumerate(labels)}

        # Invert the mapping to group nodes by community
        communities = {}
        for node, community in node_to_community.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        # Convert communities to a list of lists (if needed for subsequent processing)
        communities = list(communities.values())                            

    # Label Propagation: This algorithm uses network structure alone to guide its process and doesn't require a pre-defined objective function.
    elif community_detection == 'label_propagation':
        communities = list(label_propagation_communities(G)) 

    else:
        raise ValueError(f"Invalid community detection method '{community_detection}'.")
    
    return communities

def plotly_network_graph(G, communities):
    # Generate positions for the nodes using a layout
    pos = nx.spring_layout(G)

    # Create a list of colors for visualization
    community_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'grey']
    color_map = {}
    for idx, community in enumerate(communities):
        color = community_colors[idx % len(community_colors)]
        for node in community:
            color_map[node] = color

    # Extract node positions and colors for the plot
    x_nodes = [pos[node][0] for node in G.nodes()]  # x-coordinates of nodes
    y_nodes = [pos[node][1] for node in G.nodes()]  # y-coordinates of nodes
    node_colors = [color_map[node] for node in G.nodes()]  # node colors

    # Find the maximum similarity (edge weight) for scaling
    max_weight = max(data['weight'] for _, _, data in G.edges(data=True))

    # Create a scatter plot for each edge
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        width = edge[2]['weight'] / max_weight * 10  # Adjust scaling factor as needed
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None], 
            line=dict(width=width, color='rgba(100,100,100,0.1)'),  # Adjust color and transparency
            hoverinfo='none',
            mode='lines'
        )
        edge_traces.append(edge_trace)

    # Create a scatter plot of nodes
    node_trace = go.Scatter(
        x=x_nodes, 
        y=y_nodes, 
        mode='markers',
        marker=dict(size=10, color=node_colors, line_width=2),
        text=list(G.nodes()),  # Hover text (node ID)
        hoverinfo='text'
    )

    # Create a figure
    fig = go.Figure(data=edge_traces + [node_trace],
                 layout=go.Layout(
                    title='<br>Network graph with communities',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Show the figure
    fig.show()

# Example usage
# G = nx.Graph()
# communities = list of communities
# ... code to populate your graph ...
# plotly_network_graph(G, communities)





def extract_ratings(df, df_weights):
    """
    Extracts the ratings from the DataFrame and returns a new DataFrame with weighted ratings.
    Throws an error if a keyword in df does not have a corresponding weight in df_weights.
    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - df_weights (DataFrame): DataFrame containing the weights for each keyword.
    Returns:
    - df_ratings (DataFrame): DataFrame containing the weighted ratings.
    """
    # Columns to exclude from the ratings
    non_keyword_columns = ['id', 'clean_title', 'clean_abstract', 'summary', 'session_num', 'talk_num', 'major_branch', 'session_branch']

    # Extract keywords from the dataframe, by excluding other columns
    keywords = [col for col in df.columns if col not in non_keyword_columns]

    # Convert keyword columns to numeric and fill NaNs with 0
    df_ratings = df[keywords].apply(pd.to_numeric).fillna(0)

    # Apply weights to the ratings and check for unmatched keywords
    for keyword in keywords:
        if keyword in df_weights['keyword'].values:
            
            weight = df_weights.loc[df_weights['keyword'] == keyword, 'initial_weight'].iloc[0]
            
            # Multiply the ratings by the weight
            df_ratings[keyword] *= weight

        else:
            raise ValueError(f"Keyword '{keyword}' in df does not have a corresponding weight in df_weights.")

    # Remove all columns where every row is zero from ratings
    df_ratings = df_ratings.loc[:, (df_ratings != 0).any(axis=0)]

    # Replace zeros with NaNs and fill NaNs with the mean of each column
    df_ratings.replace(0, np.nan).fillna(df_ratings.mean())

    # Add the id column to the ratings DataFrame
    # df_ratings = df_ratings.set_index(df['id'])
    df_ratings.index = df.index

    return df_ratings