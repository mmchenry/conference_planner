import os
import pandas as pd
import markdown
import make_sessions as ms
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import math




def render_schedule_html(data_root, presentation_type, grouping_level='sessions', include_summary=False):
    """
    Generate a schedule of sessions within each major branch in HTML format from CSV files.
    """

    # Load grouping data
    df_group = pd.read_csv(os.path.join(data_root,  presentation_type + '_grouping.csv'), index_col='id')

    # Load rating data for presentation type
    df_ratings = pd.read_csv(os.path.join(data_root,  presentation_type + '_ratings.csv'), index_col='id')

    # Load abstract data for presentation type
    df = pd.read_csv(os.path.join(data_root,  presentation_type + '.csv'), index_col='id')

    # Add the major_group column to df, matching indicies
    df['major_group'] = df_group['major_group']
    df['session_num'] = df_group['session_num']
    df['summary']     = df_ratings['summary']
    df['talk_num']    = df_group['talk_num']

    # Initiate the markdown string
    markdown_str = ""

    # Listings and summary data for sessions/groups
    if grouping_level == 'sessions':
        group_list = sorted(df['session_num'].unique())
        group_summ = pd.read_csv(os.path.join(data_root,  presentation_type + '_session_summary.csv'))

    elif grouping_level == 'groups':
        group_list = sorted(df['major_group'].unique())
        group_summ = pd.read_csv(os.path.join(data_root,  presentation_type + '_group_summary.csv'))

    # Raise exception if group_list is empty
    if len(group_list) == 0:
        raise ValueError("group_list is empty")
    
    # Loop through each group
    for group in group_list:

        markdown_str += f"-------------------------------------------------------------- \n\n"

        # Get the "Top keywords" for branch match to the "Branch" column of branch_summ
        if grouping_level == 'groups':
            group_keywords = group_summ[group_summ['Group'] == group]['Top Keywords'].values[0]
        else:
            group_keywords = group_summ[group_summ['Session'] == group]['Top Keywords'].values[0]

        # Add the session/group title
        if grouping_level == 'groups':
            markdown_str += f"### Group {group}  {group_keywords}\n\n"
        elif grouping_level == 'sessions':
            markdown_str += f"### Session {group}  {group_keywords}\n\n"

        # markdown_str += f"{branch_keywords}\n\n"

        # Get the dataframe for the current session/group
        if grouping_level == 'groups':
            session_df = df[df['major_group'] == group]

        elif grouping_level == 'sessions':
            session_df = df[df['session_num'] == group]

            # if there are values for talk_num
            if len(session_df['talk_num'].unique()) > 0:
                # Sort the session dataframe by talk_num
                session_df = session_df.sort_values(by='talk_num')

        # Loop through each talk in the session
        for _, row in session_df.iterrows():

            # Abstract title
            ab_title = row['title']

            # Remove spaces at the beginning and end of ab_title
            ab_title = ab_title.strip()

            # Restate the tile in sentance case
            ab_title = ab_title[0].upper() + ab_title[1:].lower()

            if include_summary:
                # Text that includes the summary
                markdown_str += f"**{ab_title}**\n{row['summary']}\n\n"
            else:
                # Just the titles
                markdown_str += f"**{ab_title}**\n\n"

    html_str = markdown.markdown(markdown_str)
  
    # Write the output to a file   
    if grouping_level == 'groups':
        output_file = os.path.join(data_root, 'groups_' + presentation_type + '.html')
        with open(output_file, 'w') as f:
            f.write(html_str)
        print(f"Groups written to {output_file}")
    else:
        output_file = os.path.join(data_root, 'sessions_' + presentation_type + '.html')
        with open(output_file, 'w') as f:
            f.write(html_str)
        print(f"Sessions written to {output_file}")
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


def word_cloud(data_root, presentation_type, grouping_level='groups', num_columns=4):
    """
    Generate a word cloud for each session/group in a presentation type.
    
    Parameters:
    - data_root (str): Path to the directory containing the CSV files.
    - presentation_type (str): Name of the presentation type.
    - grouping_level (str): Level of grouping to use for the word cloud.
    - num_columns (int): Number of columns for the subplot.
    
    Returns:
    - None
    """


    # Load grouping data
    df_group = pd.read_csv(os.path.join(data_root,  presentation_type + '_grouping.csv'), index_col='id')

    # Load abstract data for presentation type
    df = pd.read_csv(os.path.join(data_root,  presentation_type + '.csv'), index_col='id')

    # Add the major_group column to df, matching indicies
    df['major_group'] = df_group['major_group']
    df['session_num'] = df_group['session_num']
    df['talk_num']    = df_group['talk_num']

    if grouping_level == 'groups':
        group_list = sorted(df['major_group'].unique())

    # Determine the number of rows needed for the subplot
    num_groups = len(group_list)
    
    num_rows = math.ceil(num_groups / num_columns)

    # Create a figure for the subplots
    plt.figure(figsize=(20, 5 * num_rows))

    # Loop through each group
    for i, group in enumerate(group_list, 1):
        # Get the dataframe for the current session/group
        session_df = df[df['major_group'] == group]

        # Concatenate all abstracts in the group
        text = ' '.join(session_df['clean_abstract'].dropna())

        # Create a word cloud object
        wordcloud = WordCloud(width=800, height=800, 
                              background_color='white', 
                              min_font_size=10).generate(text)

        # Add subplot for each group
        plt.subplot(num_rows, num_columns, i)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Group {group}')
        plt.axis("off")

    plt.tight_layout()
    plt.show()

