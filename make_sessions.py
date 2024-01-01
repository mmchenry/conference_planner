import os
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import string
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
import random
import string



def extract_ratings(df, df_weights, analysis_step='clustering', include_id=False):
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
            
            # For hierarchical clustering 
            if analysis_step == 'clustering':
                weight = df_weights.loc[df_weights['keyword'] == keyword, 'weight_clustering'].iloc[0]

            # For sequencing
            elif analysis_step == 'sequencing':
                weight = df_weights.loc[df_weights['keyword'] == keyword, 'weight_sequencing'].iloc[0]

            else:
                raise ValueError(f"Invalid analysis_step: {analysis_step}")
            
            # Multiply the ratings by the weight
            df_ratings[keyword] *= weight
            
            # For sequencing, exclude keywords where all clustering weights are zero
            # if analysis_step == 'sequencing' and df_weights[df_weights['keyword'] == keyword]['weight_clustering'].any():
            #     df_ratings.drop(columns=[keyword], inplace=True)

        else:
            raise ValueError(f"Keyword '{keyword}' in df does not have a corresponding weight in df_weights.")

    # Remove all columns where every row is zero from ratings
    df_ratings = df_ratings.loc[:, (df_ratings != 0).any(axis=0)]

    # Replace zeros with NaNs and fill NaNs with the mean of each column
    df_ratings.replace(0, np.nan).fillna(df_ratings.mean())

    # Add the id column to the ratings DataFrame
    if include_id:
        df_ratings['id'] = df['id']

    return df_ratings




def find_combinations(target_sum, min_value=6, max_value=8):
    """
    Returns a list of lists of integers that sum to target_sum, with each list containing integers between min_value and max_value (inclusive). If a combination of integers that sum to target_sum does not exist, one of the numbers is permitted to be less than min_value.
    
    Parameters:
    target_sum (int): The desired sum of the integers in the combinations
    min_value (int): The minimum value of each integer in the combinations
    max_value (int): The maximum value of each integer in the combinations

    Returns:
    list: A list of lists of integers that sum to target_sum, with each list containing integers between min_value and max_value (inclusive)
    """
    def generate_combinations(partial_sum, combination, used_lower=False):
        nonlocal in_range_found
        
        if partial_sum == target_sum:
            if all(min_value <= num <= max_value for num in combination):
                in_range_found = True
            results.append(combination)
            return
        if partial_sum > target_sum:
            return

        for number in range(min_value, max_value + 1):
            generate_combinations(partial_sum + number, combination + [number], used_lower)

        if not used_lower and not in_range_found:
            for number in range(1, min_value):
                generate_combinations(partial_sum + number, combination + [number], True)

    results = []
    in_range_found = False
    generate_combinations(0, [])

    # Filter out combinations with numbers outside of the desired range if in-range combinations exist
    if in_range_found:
        results = [comb for comb in results if all(min_value <= item <= max_value for item in comb)]

    # filter out duplicate combinations
    results = list(set(tuple(sorted(comb)) for comb in results))

    # Sort numbers in each set
    results = [sorted(x) for x in results]

    return results


def fuse_clusters(df, min_size):
    """
    Fuses clusters that are intermediate in size.
    
    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - min_size (int): Minimum number of abstracts per session.
    
    Returns:
    - df (DataFrame): DataFrame containing the abstracts and ratings, with cluster_num column added.
    """

    # Start by including all clusters
    branches_to_fuse = np.unique(df['major_branch'])

    # Loop until no more clusters need to be fused
    while len(branches_to_fuse) > 1:

        # Count the number of talks in each major branch
        branch_counts = df['major_branch'].value_counts()

        # Sort branch_counts by its index alphabetically
        branch_counts = branch_counts.sort_index()

        # Identify branches that need to be fused
        # branches_to_fuse = branch_counts[(branch_counts > max_size) & (branch_counts < min_size + max_size)]
        branches_to_fuse = branch_counts[branch_counts < min_size]
        
        # If no branches need fusing, exit the loop
        if branches_to_fuse.empty:
            break
            
        # Get the current branch
        curr_branch = branches_to_fuse.index[0]

        # Sort the filtered_branch_counts indices alphabetically
        sorted_indices = sorted(branch_counts.index)

        # Find the indices before and after the current branch
        prev_index = next((i for i in sorted_indices[::-1] if i < curr_branch), None)
        next_index = next((i for i in sorted_indices if i > curr_branch), None)

        # Determine the closest index based on size, prioritizing the smaller branch
        if prev_index and next_index:
            if branch_counts[prev_index] <= branch_counts[next_index]:
                closest_index = prev_index
            else:
                closest_index = next_index
        elif prev_index:
            closest_index = prev_index
        elif next_index:
            closest_index = next_index
        else:
            closest_index = None
            
        # Replace the current branch with the closest branch in df
        if closest_index is not None:
            df['major_branch'].replace({curr_branch: closest_index}, inplace=True)
            print(f"Fused cluster {curr_branch} with {closest_index}")
        else:
            # If no closest branch is found, break the loop
            break

    return df


import pandas as pd

def summarize_branch_keywords(df, df_weights, num_keywords=5, echo=True):
    """
    Summarizes the keywords for each major branch, including the mean value for each keyword.
    
    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - df_weights (DataFrame): DataFrame containing weights for analysis.
    - num_keywords (int): Number of keywords to include in the summary.
    
    Returns:
    - branch_summaries (DataFrame): DataFrame containing the summary for each branch.
    """

    # Extract ratings to get keyword columns
    ratings = extract_ratings(df, df_weights, analysis_step='clustering')
    keywords_columns = ratings.columns

    # Initialize an empty DataFrame to store branch summaries
    branch_summ = pd.DataFrame()

    # Group by major branch and calculate mean for each keyword
    branch_means = df.groupby('major_branch')[keywords_columns].mean()

    # Get the count of talks in each branch
    branch_counts = df['major_branch'].value_counts()

    # Iterate over each branch
    for branch, data in branch_means.iterrows():
        # Sort keywords by mean rating and select top N
        top_keywords_data = data.sort_values(ascending=False).head(num_keywords)
        top_keywords = [f"{kw} ({mean_val:.2f})" for kw, mean_val in top_keywords_data.items()]
        talk_count = branch_counts[branch]

        # Create a DataFrame for the current branch summary
        branch_df = pd.DataFrame({'Branch': [branch], 'Number of Talks': [talk_count], 'Top Keywords': [top_keywords]})

        # Concatenate the current branch summary to the main summary DataFrame
        branch_summ = pd.concat([branch_summ, branch_df], ignore_index=True)

    if echo:
        # Display the branch summary
        branch_summary_styled = branch_summ.style.set_properties(subset=['Top Keywords'], **{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
        display(branch_summary_styled)

    return branch_summ


def run_hierarchical(df, df_weights, min_size=16):
    """
    Runs hierarchical clustering on the DataFrame until some sessions are found in desired range of sizes.

    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - min_size (int): Minimum number of abstracts per major branch.
    - max_size (int): Maximum number of abstracts per major branch.

    Returns:
    - df (DataFrame): DataFrame containing the abstracts and ratings, with major_branch column added.
    - distance_threshold (float): Distance threshold used for clustering.
    """

    # Threshold for cutting the dendrogram
    distance_threshold = 1

    # Increment for increasing the threshold
    dist_increment = 0.1

    # Threshold proportion of valid branches
    prop_threshold = 0.5

    # Maximum distance threshold
    max_distance = 10

    # Whether this approach worked
    success = False

    # Loop until a valid number of clusters is found
    while distance_threshold<max_distance:
        try:        
            # Run hierarchical clustering
            df = hierarchical_clustering(df, df_weights, distance_threshold=distance_threshold, display_dendrogram=False)

            # Get the count of talks in each branch
            num_talks = df['major_branch'].value_counts()

            # if prop_valid>=prop_threshold:
            if np.min(num_talks) >= min_size:
                success = True
                break
            else:
                distance_threshold += dist_increment

        except:
            distance_threshold += dist_increment

    # If the hierarchical clustering didn't work
    if not success:
        # Set df[branch_type] to 'A' if it didn't work
        df['major_branch'] = 'A'

    # If the hierarchical clustering worked
    else:
        # Fuse clusters, if necessary
        df = fuse_clusters(df, min_size)

    return df, distance_threshold


def hierarchical_clustering(df, df_weights, distance_threshold, display_dendrogram=False):
    """
    Performs hierarchical clustering on the DataFrame and returns the DataFrame with a column for the major branch.

    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - distance_threshold (float): Distance threshold used for clustering.
    - display_dendrogram (bool): Whether to display the dendrogram.
    - branch_type (str): Type of branch to use for clustering ('major_branch' or 'session_branch').

    Returns:
    - df (DataFrame): DataFrame containing the abstracts and ratings, with major_branch column added.
    """

    # Extract ratings
    ratings = extract_ratings(df, df_weights, analysis_step='clustering')

    # Perform hierarchical clustering
    Z = hierarchy.linkage(ratings, method='ward')

    # Form flat clusters
    flat_clusters = fcluster(Z, t=distance_threshold, criterion='distance')

    # Assign cluster labels (A, B, C, ...)
    max_labels = len(string.ascii_uppercase)
    cluster_labels = {idx + 1: string.ascii_uppercase[idx % max_labels] for idx in range(max(flat_clusters))}
    labeled_clusters = [cluster_labels[cluster] for cluster in flat_clusters]

    # Add cluster assignments to the DataFrame
    df['major_branch'] = labeled_clusters

    if display_dendrogram:
        plot_dendrogram(df, distance_threshold)

    return df


def plot_dendrogram(df, df_weights, distance_threshold):
    """
    Plots the dendrogram for the hierarchical clustering.
    
    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - distance_threshold (float): Distance threshold used for clustering.
    """

    # Extract ratings
    ratings = extract_ratings(df, df_weights, analysis_step='clustering')

    # Perform hierarchical clustering
    Z = hierarchy.linkage(ratings, method='ward')

    # Create and show a rotated dendrogram
    plt.figure(figsize=(10, 7))
    hierarchy.dendrogram(Z, orientation='left', color_threshold=distance_threshold)
    plt.axvline(x=distance_threshold, color='r', linestyle='--')
    plt.show()


def process_each_branch(df, df_weights, min_size=6, max_size=8, echo=False):
    """
    Processes each major branch to sequentially create sessions based on talk proximity.

    Parameters:
    - df (DataFrame): DataFrame containing 'major_branch' and other talk details.
    - min_size (int): Minimum number of talks per session.
    - max_size (int): Maximum number of talks per session.

    Returns:
    - df (DataFrame): DataFrame containing 'session_num' and 'talk_num' columns.
    """

    # if there is a session_num value, set session_num to the next value
    if df['session_num'].notnull().any():
        session_num = df['session_num'].max() + 1
    else:
        # Initialize session number
        session_num = 1

    # Loop through each major branch
    for branch, group in df.groupby('major_branch'):

        # Update status
        if echo:
            print(f"Processing branch {branch} . . .")
        
        # Number of talks in group
        num_talks = len(group)

        # Find distances between talks
        df_dists = calculate_distances(extract_ratings(group, df_weights, analysis_step='sequencing'))
            
        # Indicies for set of sessions with closest distances
        set_idx = find_best_matches(group, df_dists, min_size, max_size)

        # Loop thru each session
        for sess_idx in set_idx:

            # Set the session_num for each talk in the group
            df.loc[sess_idx, 'session_num'] = session_num

            # Add talk numbers
            df.loc[sess_idx, 'talk_num'] = np.arange(1,len(sess_idx)+1)

            # Advance the session number
            session_num += 1

    return df


def find_best_matches(group, df_dists, min_size, max_size):
    """
    Finds the best ordering of talks within each session group.
    
    Parameters:
    - group (DataFrame): DataFrame containing the abstracts and ratings for a particular major branch.
    - df_dists (DataFrame): DataFrame containing the pairwise distances between talks.
    - min_size (int): Minimum number of talks per session.
    - max_size (int): Maximum number of talks per session.

    Returns:
    - best_session_groups (list): A list of lists containing the best ordering of talks within each session group.
    """

    # Number of random versions to try
    n_rand = int(2*len(group))

    # Find combinations of numbers of talks that meet the size criteria
    session_num_sets = find_combinations(len(group), min_size, max_size)

    # Initialize variables to store the best version
    best_session_groups = None
    best_score = float('inf')
    
    # Loop thru each session-number set
    for index, curr_set in enumerate(session_num_sets):

        # Try multiple random versions of curr_set
        for _ in range(n_rand):

            session_groups = []
            remaining_talks = group.index.tolist()
            session_group_scores = []

            # Loop thru each set in curr_set
            for set_size in curr_set:

                # Randomly select the seed talk index
                curr_index = random.choice(remaining_talks)
                selected_talks = []

                # Loop thru items in each set to find order of closest talks
                for _ in range(set_size):

                    # Select the talk closest to the seed index
                    sorted_distances = df_dists.loc[curr_index, remaining_talks].sort_values()
                    next_talk = sorted_distances.index[0]

                    # Add the talk to the selected talks
                    selected_talks.append(next_talk)

                    # Update the seed index
                    curr_index = next_talk

                    # Remove the talk from remaining_talks
                    remaining_talks.remove(next_talk)

                # Add the selected talks to session_groups
                session_groups.append(selected_talks)

                # Calculate the mean distance between talks in each session group
                subset = df_dists.loc[selected_talks, selected_talks]
                mean_distance = subset.mean().mean()
                session_group_scores.append(mean_distance)

            # Store the version with the smallest mean score
            current_score = np.mean(session_group_scores)
            if current_score < best_score:
                best_score = current_score
                best_session_groups = session_groups   
    
    return best_session_groups


def calculate_distances(ratings):
    """
    Calculate the Euclidean distance between each pair of talks.

    Parameters:
    - ratings (DataFrame): A pandas DataFrame with each row representing a talk
      and each column a keyword rating.

    Returns:
    - DataFrame: A pandas DataFrame representing the pairwise Euclidean distances.
    """

    

    # Compute pairwise distances
    distances = pdist(ratings, metric='euclidean')
    
    # Convert to a square form DataFrame
    distance_matrix = squareform(distances)
    
    # Convert to a square form DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=ratings.index, columns=ratings.index)
    
    # Assign NaN to the diagonal and lower triangle elements
    np.fill_diagonal(distance_df.values, np.nan)
    distance_df = distance_df.where(np.triu(np.ones(distance_df.shape), k=1).astype(np.bool_))

    return distance_df



