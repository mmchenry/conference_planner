import os
import pandas as pd
import markdown
import make_sessions as ms

def merge_dataframes(df, data_root):
    """
    Creates a dataframe with all the abstract data for a given division.
    """

    # Load all abstract data
    df_raw = pd.read_csv(os.path.join(data_root, 'abstracts_revised.csv'))

    # Select only the 'ID' and 'Abtitle' columns from df_raw, and rename 'Abtitle' to 'title'
    # df_raw_selected = df_raw[['id', 'title']].rename(columns={'AbTitle': 'title'})

    # Merge selected columns from df_raw into df on the matching IDs
    df_full = pd.merge(df, df_raw, how='left', left_on='id', right_on='id')

    # Optionally, you can drop the extra 'ID' column if it's redundant
    # df_full.drop('id', axis=1, inplace=True)

    return df_full


def list_branches_html(df, data_root, presentation_type, branch_summ, include_summary=False):
    """
    Generate a schedule of sessions within each major branch in HTML format from CSV files.
    """

    # Initiate the markdown string
    markdown_str = ""

    # Loop through each session
    for branch in sorted(df['major_branch'].unique()):

        markdown_str += f"-------------------------------------------------------------- \n\n"

        # Get the "Top keywords" for branch match to the "Branch" column of branch_summ
        branch_keywords = branch_summ[branch_summ['Branch'] == branch]['Top Keywords'].values[0]

        
        # Add the session title
        markdown_str += f"### Branch {branch}  {branch_keywords}\n\n"

        # markdown_str += f"{branch_keywords}\n\n"

        # Get the dataframe for the current session
        session_df = df[df['major_branch'] == branch]

        # Sort the session dataframe by talk_num
        session_df = session_df.sort_values(by='talk_num')

        # Loop through each talk in the session
        for _, row in session_df.iterrows():

            # Abstract title
            ab_title = row['title']

            # remove space at the end of ab_title
            if ab_title[-1] == ' ':
                ab_title = ab_title[:-1]

            # Restate the tile in sentance case
            ab_title = ab_title[0].upper() + ab_title[1:].lower()

            if include_summary:
                # Text that includes the summary
                markdown_str += f"**{ab_title}**\n{row['summary']}\n\n"
            else:
                # Just the titles
                markdown_str += f"**{ab_title}**\n\n"

    html_str = markdown.markdown(markdown_str)

    output_file = os.path.join(data_root, 'branches_' + presentation_type + '.html')

    with open(output_file, 'w') as f:
        f.write(html_str)

    print(f"Schedule written to {output_file}")
    print("Copy and paste path into a web browser")


def render_div_schedule_html(df_full, df, df_weights, data_root, presentation_type, include_summary=False, num_ratings=6, include_ratings=True):
    """
    Generate a schedule of sessions within a division in HTML format from CSV files.

    Parameters:
    - source_dir (str): Path to the directory containing the source abstracts Excel file.
    - inter_dir (str): Path to the directory containing session CSV files.
    - output_dir (str): Path to the directory for output HTML files.
    - abstract_filename (str): Name of the Excel file containing abstracts.
    - include_summary (bool): Whether to include the abstract summary in the output.
    - num_ratings (int): Number of top ratings to include in the output.
    - include_ratings (bool): Whether to include the ratings in the output.
    """

    # Initiate the markdown string
    markdown_str = ""

    # Extract the ratings
    ratings = ms.extract_ratings(df, df_weights, analysis_step='sequencing', include_id=True)

    # Loop through each session
    for session_num in sorted(df_full['session_num'].unique()):

        # Add the session title
        markdown_str += f"### Session {session_num}\n\n"

        # Get the dataframe for the current session
        session_df = df_full[df_full['session_num'] == session_num]

        # Sort the session dataframe by talk_num
        session_df = session_df.sort_values(by='talk_num')

        # Loop through each talk in the session
        for _, row in session_df.iterrows():
            
            if include_ratings:
                # Get the top num_ratings ratings for the current presentation
                ratings_str = top_ratings(ratings[ratings['id'] == row['id']], num_ratings)

            # Abstract title
            ab_title = row['title']

            # remove space at the end of ab_title
            if ab_title[-1] == ' ':
                ab_title = ab_title[:-1]

            # Remove space at the end of ab_title
            ab_title = ab_title.rstrip()

            # Restate the tile in sentence case
            ab_title = ab_title[0].upper() + ab_title[1:].lower()

            # Add summarye to the markdown string
            if include_summary:
                # Text that includes the summary
                markdown_str += f"**{ab_title}**\n{row['summary']}\n"
            else:
                # Just the titles
                markdown_str += f"**{ab_title}**\n"

            if include_ratings:
                # Add the ratings to the output
                markdown_str += f"{ratings_str}\n\n"

    html_str = markdown.markdown(markdown_str)

    output_file = os.path.join(data_root, presentation_type + '.html')

    with open(output_file, 'w') as f:
        f.write(html_str)

    print(f"Schedule written to {output_file}")
    print("Copy and paste path into a web browser")

def top_ratings(curr_ratings, num_ratings=4):
    """
    Get the top N ratings for a given presentation.
    
    Parameters:
    - curr_ratings (pd.DataFrame): DataFrame of ratings for a single presentation.
    - num_ratings (int): Number of top ratings to return.
    
    Returns:
    - formatted_ratings (list): List of strings with rating names and values.
    """


    # Drop the id column
    curr_ratings = curr_ratings.drop(columns=['id'])

    # Transpose the DataFrame
    curr_ratings_transposed = curr_ratings.T

    # Rename the column for clarity
    curr_ratings_transposed.columns = ['rating']

    # Sort by the rating values
    sorted_ratings = curr_ratings_transposed.sort_values(by='rating', ascending=False)

    # If you want to get the top N ratings
    top_ratings = sorted_ratings.head(num_ratings)

    # If you need to convert this back to the original format (with ratings as columns)
    top_ratings_transposed = top_ratings.T

    # Format the output as a list of strings with rating names and values
    formatted_ratings = [f"{rating} ({value:.2f})" for rating, value in top_ratings['rating'].items()]

    return formatted_ratings


def render_meeting_schedule_html(df_full, division, presentation_type, output_dir, include_summary=False):
    """
    Generate a schedule of sessions for entire meeting in HTML format, from divisional CSV files.

    Parameters:
    - source_dir (str): Path to the directory containing the source abstracts Excel file.
    - inter_dir (str): Path to the directory containing session CSV files.
    - output_dir (str): Path to the directory for output HTML files.
    - abstract_filename (str): Name of the Excel file containing abstracts.
    """
# TODO