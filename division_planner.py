import os
import pandas as pd
import markdown

def load_div_dataframe(df, source_dir, abstract_filename):
    """
    Creates a dataframe with all the abstract data for a given division.
    """

    # Load all abstract data
    df_raw = pd.read_excel(os.path.join(source_dir, abstract_filename + '.xlsx'))

    # Select only the 'ID' and 'Abtitle' columns from df_raw, and rename 'Abtitle' to 'title'
    df_raw_selected = df_raw[['ID', 'AbTitle']].rename(columns={'AbTitle': 'title'})

    # Merge selected columns from df_raw into df on the matching IDs
    df_full = pd.merge(df, df_raw_selected, how='left', left_on='id', right_on='ID')

    # Optionally, you can drop the extra 'ID' column if it's redundant
    df_full.drop('ID', axis=1, inplace=True)

    return df_full




def render_div_schedule_html(df_full, division, presentation_type, output_dir, include_summary=False):
    """
    Generate a schedule of sessions in HTML format from CSV files.

    Parameters:
    - source_dir (str): Path to the directory containing the source abstracts Excel file.
    - inter_dir (str): Path to the directory containing session CSV files.
    - output_dir (str): Path to the directory for output HTML files.
    - abstract_filename (str): Name of the Excel file containing abstracts.
    """

    # Path to the session CSV files
    curr_out_dir = os.path.join(output_dir, presentation_type + '_html')

    # Ensure output directory exists
    if not os.path.exists(curr_out_dir):
        os.makedirs(curr_out_dir)

    # Initiate the markdown string
    markdown_str = ""

    # markdown_str += f"## {csv_file.replace('.csv', '')}\n\n"

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

    output_file = os.path.join(curr_out_dir, division + '.html')

    with open(output_file, 'w') as f:
        f.write(html_str)

    print(f"Schedule written to {output_file}")
    print("Copy and paste path into a web browser")