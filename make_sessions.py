import os
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import string
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
import random
import string
# import network_analysis as na



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
    non_keyword_columns = ['id', 'clean_title', 'clean_abstract', 'summary', 'session_num', 'talk_num', 'major_group', 'session_branch']

    # Extract keywords from the dataframe, by excluding other columns
    keywords = [col for col in df.columns if col not in non_keyword_columns]

    # Convert keyword columns to numeric and fill NaNs with 0
    df_ratings = df[keywords].apply(pd.to_numeric).fillna(0)

    # Apply weights to the ratings and check for unmatched keywords
    for keyword in keywords:
        if keyword in df_weights['keyword'].values:
            
            # For hierarchical clustering 
            if analysis_step == 'clustering':
                weight = df_weights.loc[df_weights['keyword'] == keyword, 'group_weights'].iloc[0]

            # For sequencing
            elif analysis_step == 'sequencing':
                weight = df_weights.loc[df_weights['keyword'] == keyword, 'session_weights'].iloc[0]

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


def is_divisible_by_combinations(number, current_sum=0, max_sum=None):
    """
    Determines whether a number is divisible by the sum of any combination of 6, 7, and 8.

    Parameters:
    - number (int): The number to check.
    - current_sum (int): The current sum of the digits.
    - max_sum (int): The maximum sum of the digits.

    Returns:
    - bool: Whether the number is divisible by the sum of any combination of 6, 7, and 8.
    """

    digits=[6, 7, 8]
    
    if max_sum is None:
        max_sum = number

    # Base case: if current_sum exceeds max_sum, stop the recursion
    if current_sum > max_sum:
        return False

    # Check if the current sum divides the number
    if current_sum != 0 and number % current_sum == 0:
        return True

    # Recursive case: try adding each digit to the current sum and recurse
    for digit in digits:
        if is_divisible_by_combinations(number, current_sum + digit, max_sum):
            return True

    return False

def check_divisibility(numbers):
    """
    Tests whether each number in a list is divisible by the sum of any combination of 6, 7, and 8 using is_divisible_by_combinations().
    """
    results = {}
    for number in numbers:
        results[number] = is_divisible_by_combinations(number)
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
    branches_to_fuse = np.unique(df['major_group'])

    # Loop until no more clusters need to be fused
    while len(branches_to_fuse) > 1:

        # Count the number of talks in each major branch
        branch_counts = df['major_group'].value_counts()

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
            df['major_group'].replace({curr_branch: closest_index}, inplace=True)
            print(f"Fused cluster {curr_branch} with {closest_index}")
        else:
            # If no closest branch is found, break the loop
            break

    return df


def summarize_group_keywords(data_root, presentation_type, grouping_level='groups', num_keywords=5, echo=True):
    """
    Summarizes the keywords for each major branch, including the mean value for each keyword.
    
    Parameters:
    - data_root (str): Path to the data folder.
    - presentation_type (str): Type of presentation ('talks' or 'posters').
    - grouping_level (str): Level of grouping ('group' or 'session').
    - num_keywords (int): Number of keywords to include in the summary.
    - echo (bool): Whether to display the summary.
    
    Returns:
    - branch_summaries (DataFrame): DataFrame containing the summary for each branch.
    """

    # Load grouping data
    df_group = pd.read_csv(os.path.join(data_root,  presentation_type + '_grouping.csv'), index_col='id')

    # Load abstract data for presentation type
    df = pd.read_csv(os.path.join(data_root,  presentation_type + '_ratings.csv'), index_col='id')

    # Add the major_group column to df, matching indicies
    df['major_group'] = df_group['major_group']
    df['session_num'] = df_group['session_num']
    df['talk_num']    = df_group['talk_num']
    
    # Load keyword weights
    df_weights = pd.read_excel(os.path.join(data_root, 'keyword_weights.xlsx')) 

    # Extract ratings to get keyword columns
    ratings = extract_ratings(df, df_weights, analysis_step='clustering')
    keywords_columns = ratings.columns

    # Initialize an empty DataFrame to store branch summaries
    group_summ = pd.DataFrame()

    # Group by major branch and calculate mean for each keyword
    if grouping_level=='groups':
        branch_means = df.groupby('major_group')[keywords_columns].mean()
        # Get the count of talks in each branch
        branch_counts = df['major_group'].value_counts()

    elif grouping_level=='session':
        branch_means = df.groupby('session_num')[keywords_columns].mean()
         # Get the count of talks in each session
        branch_counts = df['session_num'].value_counts()
    else:
        raise ValueError(f"Invalid grouping_level: {grouping_level}")

    if len(branch_means) == 0:
        raise ValueError("branch_means is empty")
    
    # Iterate over each branch
    for branch, data in branch_means.iterrows():

        # Sort keywords by mean rating and select top N
        top_keywords_data = data.sort_values(ascending=False).head(num_keywords)
        top_keywords = [f"{kw} ({mean_val:.2f})" for kw, mean_val in top_keywords_data.items()]
        talk_count = branch_counts[branch]

        # Create a DataFrame for the current branch summary
        if grouping_level=='groups':
            branch_df = pd.DataFrame({'Group': [branch], 'Number of Talks': [talk_count], 'Top Keywords': [top_keywords]})
        else:
            branch_df = pd.DataFrame({'Session': [branch], 'Number of Talks': [talk_count], 'Top Keywords': [top_keywords]})

        # Concatenate the current branch summary to the main summary DataFrame
        group_summ = pd.concat([group_summ, branch_df], ignore_index=True)

    if echo:
        # Display the branch summary
        branch_summary_styled = group_summ.style.set_properties(subset=['Top Keywords'], **{'text-align': 'left'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
        display(branch_summary_styled)

    # File suffix
    if grouping_level=='groups':
        file_suffix = '_group_summary.csv'
    else:
        file_suffix = '_session_summary.csv'

    # Write the branch summary to a csv file
    group_summ.to_csv(os.path.join(data_root,  presentation_type + file_suffix), index=False)
    print(f"Saved branch summary to {os.path.join(data_root,  presentation_type + '_group_summary.csv')}")



def run_hierarchical(data_root, presentation_type='talks', data_type='similarity', min_size=12):
    """
    Runs hierarchical clustering on the DataFrame until some sessions are found in desired range of sizes. Saves a csv file ('_grouping.csv') with the major_group assignments.

    Parameters:
    - data_root (str): Path to the data folder.
    - presentation_type (str): Type of presentation ('talks' or 'posters').
    - data_type (str): Type of data to use for clustering ('similarity' or 'keywords').
    - min_size (int): Minimum number of abstracts per major branch.

    Returns:
    - distance_threshold (float): Distance threshold used for clustering.
    """

    # Path for the output file
    out_path = os.path.join(data_root,  presentation_type + '_grouping.csv')

    # Threshold for cutting the dendrogram
    distance_threshold = 1

    # Increment for increasing the threshold
    dist_increment = 0.1

    # Threshold proportion of valid branches
    prop_threshold = 0.5

    # Maximum distance threshold
    max_distance = 10

    # Variable needed to score whether this approach worked
    success = False

    # Loop until a valid number of clusters is found
    while distance_threshold<max_distance:
        try:        
            # Run hierarchical clustering
            Z, df = hierarchical_clustering(data_root,  distance_threshold=distance_threshold, 
                                         presentation_type=presentation_type, data_type=data_type, 
                                         display_dendrogram=False)

            # Get the count of talks in each branch
            num_talks = df['major_group'].value_counts()

            # if prop_valid>=prop_threshold:
            if (np.min(num_talks) >= min_size):
                if (presentation_type=='talks'):
                    # Check that group is divisible by 6, 7, or 8
                    divisibility = check_divisibility(num_talks)

                    if all(divisibility.values()):
                        success = True
                        break
                    else:
                        distance_threshold += dist_increment
                else:
                    success = True
                    break
            else:
                distance_threshold += dist_increment

        except:
            distance_threshold += dist_increment

    # If the hierarchical clustering didn't work
    if not success:
        # Set df[branch_type] to 'A' if it didn't work
        df['major_group'] = 'A'
        print(f"Warning: Unable to find a valid number of clusters for {presentation_type}.")

    # If the hierarchical clustering worked
    else:
        # Fuse clusters, if necessary
        df = fuse_clusters(df, min_size)

    # Create new dataframe, df_group, with the id and major_group columns, add columns for session_num and talk_num
    df_group = df[['major_group']].copy()
    df_group['session_num'] = np.nan
    df_group['talk_num'] = np.nan

    # Write updated dataframe
    df_group.to_csv(out_path, index=True) 
    print(f"Saved {presentation_type} data with major_group assignments to {out_path}")   
    
    return Z, distance_threshold


def hierarchical_clustering(data_root, distance_threshold, presentation_type='talks', 
                            data_type='similarity', display_dendrogram=False):
    """
    Performs hierarchical clustering on the DataFrame and returns the DataFrame with a column for the major branch.

    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - distance_threshold (float): Distance threshold used for clustering.
    - display_dendrogram (bool): Whether to display the dendrogram.
    - branch_type (str): Type of branch to use for clustering ('major_group' or 'session_branch').

    Returns:
    - df (DataFrame): DataFrame containing the abstracts and ratings, with major_group column added.
    """

    # Load abstract data for presentation type
    df = pd.read_csv(os.path.join(data_root,  presentation_type + '_ratings.csv'), index_col='id')

    # Clustering based on similarity
    if data_type == 'similarity':
        # Load similarity matrix
        df_sim = pd.read_csv(os.path.join(data_root, presentation_type + '_similarity_group.csv'), index_col='id')

        # Convert similarity to distance (assuming similarity is between 0 and 1)
        distance_matrix = df_sim.max().max() + 0.000001 - df_sim

        # Convert the distance matrix to a condensed distance matrix
        # We only need the upper triangle of the matrix, excluding the diagonal
        condensed_distance_matrix = distance_matrix.where(np.triu(np.ones(distance_matrix.shape), k=1).astype(bool)).stack()

        # Perform hierarchical clustering
        Z = hierarchy.linkage(condensed_distance_matrix, method='ward')

    # Clustering, based on keywords
    elif data_type == 'keywords':
        # Load keyword weights
        df_weights = pd.read_excel(os.path.join(data_root, 'keyword_weights.xlsx')) 

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
    df['major_group'] = labeled_clusters

    if display_dendrogram:
        plot_dendrogram(Z, distance_threshold)

    return Z, df


def plot_dendrogram(Z, distance_threshold=None):
    """
    Plots the dendrogram for the hierarchical clustering.
    
    Parameters:
    - df (DataFrame): DataFrame containing the abstracts and ratings.
    - distance_threshold (float): Distance threshold used for clustering.
    """

    # Create and show a rotated dendrogram
    plt.figure(figsize=(10, 7))
    hierarchy.dendrogram(Z, orientation='left', color_threshold=distance_threshold)
    if distance_threshold is not None:
        plt.axvline(x=distance_threshold, color='r', linestyle='--')
    plt.show()


def make_sessions(data_root, presentation_type='talks', data_type='similarity', min_size=6, max_size=8, echo=False):
    """
    Processes each major branch to sequentially create sessions based on talk proximity.

    Parameters:
    - data_root (str): Path to the data folder.
    - presentation_type (str): Type of presentation ('talks' or 'posters').
    - min_size (int): Minimum number of talks per session.
    - max_size (int): Maximum number of talks per session.

    Returns:
    - df (DataFrame): DataFrame containing 'session_num' and 'talk_num' columns.
    """

    # Load keyword weights
    if data_type == 'similarity':
        df_sim = pd.read_csv(os.path.join(data_root, presentation_type + '_similarity_session.csv'), index_col='id')     
        # Convert the columns of df_sim to integers
        df_sim.columns = df_sim.columns.astype(int) 
        
    elif data_type == 'keywords':
        df_weights = pd.read_excel(os.path.join(data_root, 'keyword_weights.xlsx'))
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    # Load abstract data for presentation type
    df = pd.read_csv(os.path.join(data_root,  presentation_type + '_ratings.csv'), index_col='id')

    # Load grouping data
    group_path = os.path.join(data_root,  presentation_type + '_grouping.csv')
    df_group   = pd.read_csv(group_path, index_col='id')

    # if there is a session_num value, set session_num to the next value
    if df_group['session_num'].notnull().any():
        session_num = df_group['session_num'].max() + 1
    else:
        # Initialize session number
        session_num = 1

    # Loop through each major branch
    for branch, group in df_group.groupby('major_group'):

        # Update status
        if echo:
            print(f"Processing group {branch} . . .")
        
        # Number of talks in group
        # num_talks = len(group)
            
        # Indicies for set of sessions with closest distances
        if data_type == 'similarity':
            set_idx = find_best_matches(group, data_type, min_size, max_size, df_sim=df_sim)

        elif data_type == 'keywords':
            set_idx = find_best_matches(group, data_type, min_size, max_size, df_weights=df_weights)

        # Loop thru each session
        for sess_idx in set_idx:

            # Set the session_num for each talk in the group
            df_group.loc[sess_idx, 'session_num'] = str(int(session_num))

            # Add talk numbers
            df_group.loc[sess_idx, 'talk_num'] = np.arange(1, len(sess_idx) + 1)

            # Convert 'talk_num' to strings formatted as integers, handling NaNs and non-numeric types
            df_group['talk_num'] = df_group['talk_num'].apply(lambda x: '{:.0f}'.format(x) if pd.notna(x) and np.issubdtype(type(x), np.number) else x)

            # Advance the session number
            session_num += 1

    # Save the dataframe to a csv file
    df_group.to_csv(group_path, index=True)
    print(f"Saved {presentation_type} with session numbers to {group_path}")


def find_best_matches(group, data_type, min_size, max_size, df_weights=None, df_sim=None):
    """
    Finds the best ordering of talks within each session group.
    
    Parameters:
    - group (DataFrame): DataFrame containing the abstracts and ratings for a particular major branch.
    - data_type (str): Type of data to use for clustering ('similarity' or 'keywords').
    - df_weights (DataFrame): DataFrame containing the weights of keywords.
    - df_sim (DataFrame): DataFrame containing the pairwise similarities between talks.
    - min_size (int): Minimum number of talks per session.
    - max_size (int): Maximum number of talks per session.

    Returns:
    - best_session_groups (list): A list of lists containing the best ordering of talks within each session group.
    """

    # Check that appropriate data is provided
    if data_type=='similarity' and df_sim is None:
        raise ValueError("df_sim must be provided if data_type is 'similarity'.")
    elif data_type=='keywords' and df_weights is None:
        raise ValueError("df_weights must be provided if data_type is 'keywords'.")

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
                
                if len(remaining_talks) < set_size:
                    print("Not enough remaining talks to form a session group.")
                    break

                # Randomly select the seed talk index
                curr_index = random.choice(remaining_talks)
                selected_talks = []

                # Find distances between talks
                if data_type == 'keywords':
                    df_dists = calculate_distances(extract_ratings(group, df_weights, analysis_step='sequencing'))

                # Loop thru items in each set to find order of closest talks
                for _ in range(set_size):

                    # Select the talk closest to the seed index using similarity data
                    if data_type == 'similarity':
                        valid_talks = [talk for talk in remaining_talks if talk in df_sim.columns]
                        if not valid_talks:
                            print("No valid talks remaining.")
                            break

                        sorted_similarities = df_sim.loc[curr_index, valid_talks].sort_values(ascending=False)
                        if sorted_similarities.empty:
                            print("sorted_similarities is empty.")
                            break

                        next_talk = sorted_similarities.index[0]

                    # Select the talk closest to the seed index using keyword data
                    elif data_type == 'keywords':
                        sorted_distances = df_dists.loc[curr_index, remaining_talks].sort_values()
                        next_talk = sorted_distances.index[0]
                    else:
                        raise ValueError(f"Invalid data_type: {data_type}")
                        
                    # next_talk = sorted_distances.index[0]

                    # Add the talk to the selected talks
                    selected_talks.append(next_talk)

                    # Update the seed index
                    curr_index = next_talk

                    # Remove the talk from remaining_talks
                    remaining_talks.remove(next_talk)

                if selected_talks:
                    # Add the selected talks to session_groups
                    session_groups.append(selected_talks)

                if data_type == 'keywords':
                    # Calculate the mean distance between talks in each session group
                    subset = df_dists.loc[selected_talks, selected_talks]
                else:
                    # Calculate the mean difference between talks in each session group
                    subset = 1-df_sim.loc[selected_talks, selected_talks]

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


def merge_dataframes(data_root, presentation_type):
    """
    Creates a dataframe (save as '_complete.csv') with all the abstract data for a given division 
    from the _ratings.csv file and abstracts_revised.csv file.

    Parameters:
    - data_root (str): Path to the data folder.
    - presentation_type (str): Type of presentation ('talks' or 'posters').

    """

    # Load abstract data for presentation type
    in_path = os.path.join(data_root,  presentation_type + '_ratings.csv')
    df = pd.read_csv(in_path)

    # Load all abstract data
    df_raw = pd.read_csv(os.path.join(data_root, 'abstracts_revised.csv'))

    # Select only the 'ID' and 'Abtitle' columns from df_raw, and rename 'Abtitle' to 'title'
    # df_raw_selected = df_raw[['id', 'title']].rename(columns={'AbTitle': 'title'})

    # Merge selected columns from df_raw into df on the matching IDs
    df_full = pd.merge(df, df_raw, how='left', left_on='id', right_on='id')

    # write the dataframe to a csv file
    df_full.to_csv(os.path.join(data_root,  presentation_type + '_complete.csv'), index=False)
    print(f"Saved {presentation_type} with all abstract data to {os.path.join(data_root,  presentation_type + '_complete.csv')}")

    # Optionally, you can drop the extra 'ID' column if it's redundant
    # df_full.drop('id', axis=1, inplace=True)


