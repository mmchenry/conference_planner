"""
Functions for preprocessing abstract and title text and adding taxonomy information to abstracts.
"""

import pandas as pd
import os
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import warnings
# import numpy as np
# from spacy import load
import re

# Need to run these lines before using NLTK to download data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')  # Add this line

# Path to revised version of the abstract data
ab_file = 'abstracts_revised.xlsx'

# Name of directory to store division files
div_dir_name = 'division_files'

# List of divisions
divisions_list = ['dab', 'dob', 'dvm', 'dcb', 'dce', 'dcpb', 'dedb','dede', 
                  'dee', 'diz', 'dnnsb', 'dpcb','edu']


def setup_directories(data_root, abstract_filename):
    """
    Function to create directories for each division and check that the abstract and keywords files exists.
    
    Parameters:
    - data_root (str): Root directory of data.
    - abstract_filename (str): Name of the Excel file containing the abstracts.
    
    Returns:
    - None: The function creates directories for each division and checks that the abstract and keywords files exist.
    """

    # Check that root paths exist
    if not os.path.exists(data_root):
        raise ValueError(f'Data directory does not exist: {data_root}')
    
    # Check that abstracts file exists
    ab_file = os.path.join(data_root, abstract_filename + '.xlsx')
    if not os.path.exists(ab_file):
        raise ValueError(f'Abstracts file does not exist: {ab_file}')
    
    # Check that keywords file exists
    kw_file = os.path.join(data_root, 'keywords.xlsx')
    if not os.path.exists(kw_file):
        warnings.warn(f'Keywords file does not exist: {kw_file}')

    # Create directory for non-divisional files
    nondiv_dir = os.path.join(data_root, 'non-division')
    if not os.path.exists(nondiv_dir):
        os.makedirs(nondiv_dir)
        print(f'Created directory {nondiv_dir}')

    # Loop thru each division
    for division in divisions_list:

        # Fuse dcb and dvm into one directory
        if division == 'dcb' or division == 'dvm':
            currdir = os.path.join(data_root, div_dir_name, 'dcb_dvm')
        
        # Otherwise, use the division name
        else:
            currdir = os.path.join(data_root, div_dir_name, division)

        # Make directory if it doesn't exist
        if not os.path.exists(currdir):
            os.makedirs(currdir)
            print(f'Created directory {currdir}')


def flag_duplicates(data_root):
    """
    Function to flag abstracts that should be excluded from analysis.

    Parameters:
    - source_data_dir (str): Root directory of data.
    - mtg_year (int): Year of the meeting for which the abstracts are collected.
    - abstract_filename (str): Name of the Excel file containing the abstracts.

    Output:
    - Saves an Excel file with a new column 'exclude' that is 1 for abstracts that should be excluded.
    - Saves an Excel file with a list of duplicate abstracts.
    """
    
    # Define file paths
    input_path       = os.path.join(data_root, ab_file)
    output_path      = input_path
    duplicates_path  = os.path.join(data_root, 'duplicates.xlsx')

    # Read Excel file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_raw = pd.read_excel(input_path)

    # Initialize exclusion columns
    df_raw['exclude'] = 0
    df_raw['exclusion_note'] = ''

    print(' ')
    print('DUPLICATE TITLES AND ID NUMBERS -------------- ')

    # Find duplicate IDs and Titles
    duplicate_ids = df_raw[df_raw.duplicated(subset='id', keep=False)]
    duplicate_titles = df_raw[df_raw.duplicated(subset='title', keep=False)]
    
    # Create dataframe to store rows with duplicate IDs or Titles
    df_dup = pd.concat([duplicate_ids, duplicate_titles]).drop_duplicates()

    # Handle cases with duplicate IDs
    for ID, group in duplicate_ids.groupby('id'):
        last_row = group.index[-1]
        df_raw.loc[group.index, 'exclude'] = 1
        df_raw.loc[last_row, 'exclude'] = 0

        if group['title'].nunique() > 1:
            df_raw.loc[group.index, 'exclusion_note'] = 'duplicate id'
            print(f"ID {id} has multiple titles:")
            for title in group['title'].unique().tolist():
                print(f"   {title}")

    # Handle cases with duplicate Titles
    for AbTitle, group in duplicate_titles.groupby('title'):
        last_row = group.index[-1]
        df_raw.loc[group.index, 'exclude'] = 1
        df_raw.loc[last_row, 'exclude'] = 0

        # If there is more than one ID
        if group['id'].nunique() > 1:
            df_raw.loc[group.index, 'exclusion_note'] = 'duplicate title'
            print(f"Multiple IDs with same title: {AbTitle}")
            for id_val in group['id'].unique().tolist():
                print(f"   {id_val}")

        # If there is only one ID
        else:
            df_raw.loc[group.index, 'exclusion_note'] = 'duplicate id and title'

    # Reorder columns to have 'exclude' right after 'ID' and 'exclusion_note' after 'exclude'
    cols = df_raw.columns.tolist()
    cols.remove('exclude')
    cols.remove('exclusion_note')
    cols.insert(cols.index('id') + 1, 'exclude')
    cols.insert(cols.index('exclude') + 1, 'exclusion_note')
    df_raw = df_raw[cols]

    # Remove those columns from the duplicate list
    cols = df_dup.columns.tolist()
    cols.remove('exclude')
    cols.remove('exclusion_note')
    df_dup = df_dup[cols]


    print(' ')

    # Save processed dataframes, only if the file does not exist
    df_raw.to_excel(output_path, index=False)
    print(f"Saved modified abstracts file: {output_path}")
    df_dup.to_excel(duplicates_path, index=False)
    print(f"Saved listing of duplicates: {duplicates_path}")


def find_duplicate_authors(data_root):
    """
    Function to find duplicate "Primary Contact - ContactID" under specific conditions.

    Parameters:
    source_data_dir (str): Root directory of data.
    mtg_year (int): Year of the meeting for which the abstracts are collected.
    inclusive_abstract_filename (str): Name of the Excel file to read from.
    """
    
    # Define file paths
    input_path   = os.path.join(data_root, 'abstracts_revised.xlsx')
    output_path  = os.path.join(data_root, 'duplicate_primary_contacts.xlsx')

    # Read Excel file
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_inclusive = pd.read_excel(input_path)

    # Filter rows where exclude==0
    df_filtered = df_inclusive[df_inclusive['exclude'] == 0]

    # Define Session Types to filter
    target_session_types = ['Contributed Poster Presentations', 'Contributed Talk Presentations']

    # Filter rows based on target Session Types
    df_filtered = df_filtered[df_filtered['Session Type'].isin(target_session_types)]

    # Find duplicate "Primary Contact - ContactID"
    duplicate_contact_ids = df_filtered[df_filtered.duplicated(subset='Primary Contact - ContactID', keep=False)]

    print(' ')
    print('DUPLICATE AUTHORS --------------------- ')

    # Print information for each duplicate case
    for contact_id, group in duplicate_contact_ids.groupby('Primary Contact - ContactID'):
        types_count = group['Session Type'].value_counts()
        if len(types_count) > 1:
            print(f"Duplicate 'ContactID': {contact_id}")
            for idx, row in group.iterrows():
                print(f"   ID: {row['id']}, Session Type: {row['Session Type']}")

    duplicate_contact_ids.to_excel(output_path, index=False)
    print(f"Saved duplicate primary contact: {output_path}")


def clean_abstracts(df):
    """
    This function preprocesses abstracts and titles by removing punctuation, lemmatizing words,
    removing common stop words, and filtering by parts of speech.
    Input:
        df: Dataframe of abstract data
    Output:
        Saves an Excel file with preprocessed abstracts in 'clean_abstract' and preprocessed titles in 'clean_title'
    """

    # Initialize a lemmatizer for text preprocessing
    lemmatizer = WordNetLemmatizer()

    # Define a function to filter out unwanted parts of speech
    def filter_words(word_list):
        pos_tags = pos_tag(word_list)
        allowed_pos = ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
        return [word for word, pos in pos_tags if pos in allowed_pos]

    clean_abstracts = []
    clean_titles = []
    stop_words = set(stopwords.words('english'))
    
    for abstract, title in zip(df['Abstract'], df['AbTitle']):
        
        if pd.isna(abstract):
            clean_abstracts.append(None)
        else:
            abstract_words = word_tokenize(str(abstract))
            abstract_words = [word.lower() for word in abstract_words if word.isalnum()]
            abstract_words = [lemmatizer.lemmatize(word) for word in abstract_words if word not in stop_words]
            abstract_words = filter_words(abstract_words)
            clean_abstracts.append(" ".join(abstract_words))
        
        if pd.isna(title):
            clean_titles.append(None)
        else:
            title_words = word_tokenize(str(title))
            title_words = [word.lower() for word in title_words if word.isalnum()]
            title_words = [lemmatizer.lemmatize(word) for word in title_words if word not in stop_words]
            title_words = filter_words(title_words)
            clean_titles.append(" ".join(title_words))

    df['clean_abstract'] = clean_abstracts
    df['clean_title']    = clean_titles
    
    return df

def process_abstracts(data_root, abstract_filename):
    """
    Extract division and topic information from abstracts and save updated abstracts file.
    """

    # dict of column names from df_raw to rename to other dataframes
    col_dict = {'ID':'id', 
                'AbTitle': 'title',
                'Abstract':'abstract',
                'clean_abstract':'clean_abstract',
                'clean_title':'clean_title'}
    
    # Path to abstract xlsx file
    input_file = os.path.join(data_root, abstract_filename + '.xlsx')

    # Read Excel file into a Pandas DataFrame
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_raw = pd.read_excel(input_file) 

    # Generate clean abstracts and titles
    df_raw = clean_abstracts(df_raw)

    # Rename the columns of df_raw
    df_raw.rename(columns=col_dict, inplace=True)

    # Define a function to extract the division for the abstract
    def div_directory(row):
        text = row['Select the best divisional affiliation for this abstract']
        if pd.isna(text):
            div_dir = None
        elif text == "Outreach, Education, and Policy":
            div_dir = "edu"
        elif '-' in text:
            div_text = text.split('-')[0].strip().lower()
            if div_text=='dvm' or div_text=='dcb':
                div_dir = 'dcb_dvm'
            else:
                div_dir = div_text
        else:
            div_dir = None
        
        return div_dir

    # Define divisional affiliation for each abstract
    df_raw['ab_division'] = df_raw.apply(div_directory, axis=1)

    # List of columns starting with "Select Topic:"
    topic_columns = [col for col in df_raw.columns if col.startswith("Select Topic:")]

    # Define a function to extract text after the dash or colon
    def extract_topic(row):
        topic_found = False  # Flag to check if a topic was found in any column
        for col in topic_columns:
            value = row[col]
            if pd.notna(value):  # Check for NaN values
                topic = value
                topic_found = True  # Set the flag to True when a topic is found
                break  # Exit the loop once a topic is found
        if topic_found:
            return topic
        else:
            return "none"  # Default value when no topic is found in any column
        
    # Apply the function to create the 'topic' column
    df_raw['topic'] = df_raw.apply(extract_topic, axis=1)

    #  Save updated df_raw file
    output_path = os.path.join(data_root, ab_file)
    df_raw.to_excel(output_path, index=False)
    print(f"Saved updated abstracts file: {output_path}")



def distribute_abstracts(data_root):
    """
    This function loads the abstract data from an xlsx file, extracts division and topic information, and saves different session types into separate csv files.
    Input:
        - data_root: Path to root directory of data

    Output:
        - Saves csv files for each division and session type
    """

    print('Distributing abstracts to CSV files:')

    # Columns to include in output dataframes
    columns_to_keep = ["id", "title", "abstract", "clean_abstract", "clean_title", "ab_division", "topic"]

    # Path to abstract file
    input_file = os.path.join(data_root, ab_file)

    # Read Excel file into a Pandas DataFrame
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        df_raw = pd.read_excel(input_file)   

    # Include only rows where exclude==0
    df_raw = df_raw[df_raw['exclude'] == 0]

    # Accounting for exported abstracts
    df_raw['exported_to_csv'] = 0

    # Count number of abstracts
    num_abs = df_raw.shape[0]

    # Initialize a counter for saved abstracts
    saved_abs_counter = 0

    # Create dataframes for non-divisional sessions
    df_special = df_raw.loc[df_raw['Session Type'] == 
                            'Special Session', columns_to_keep]
    df_none = df_raw.loc[pd.isna(df_raw['Session Type']), columns_to_keep]
    df_sym = df_raw.loc[df_raw['Session Type'] == 
                            'Symposia and complementary sessions', columns_to_keep]

    # Reindex the DataFrames
    df_special.reset_index(drop=True, inplace=True)
    df_sym.reset_index(drop=True, inplace=True)
    df_none.reset_index(drop=True, inplace=True)

    # Save non-divisional dataframes to CSV files
    nondiv_dir = os.path.join(data_root, 'non-division')
    df_special.to_csv(os.path.join(nondiv_dir, 'special.csv'), index=False)
    df_raw.loc[df_raw['Session Type'] == 'Special Session', 'exported_to_csv'] = 1
    print(f"  Saved special.csv to {nondiv_dir}")

    df_none.to_csv(os.path.join(nondiv_dir, 'none.csv'), index=False)
    df_raw.loc[pd.isna(df_raw['Session Type']), 'exported_to_csv'] = 1
    print(f"  Saved none.csv to {nondiv_dir}")

    df_sym.to_csv(os.path.join(nondiv_dir, 'symposia.csv'), index=False)
    df_raw.loc[df_raw['Session Type'] == 'Symposia and complementary sessions', 'exported_to_csv'] = 1
    print(f"  Saved symposia.csv to {nondiv_dir}")

    # Update saved_abs_counter after saving each DataFrame
    saved_abs_counter += df_special.shape[0]
    saved_abs_counter += df_none.shape[0]
    saved_abs_counter += df_sym.shape[0]

    print('  Divisional files:')

    # Loop through each division
    for division in divisions_list:

        if division == 'dcb':
            # Directory to save output files
            div_dir = os.path.join(data_root, div_dir_name, 'dcb_dvm')

            # Index of rows where 'ab_division' is 'dcb' or 'dvm'
            ab_idx =  (df_raw['ab_division'] == 'dcb_dvm') 

        elif division == 'dvm':
            continue
        else:
            # Directory to save output files
            div_dir = os.path.join(data_root, div_dir_name, division)

            # Index of rows where 'ab_division' is 'division'
            ab_idx =  df_raw['ab_division'] == division

        # Filter rows where 'Session Type' is 'Contributed Talk Presentations' and keep only specified columns
        df_con = df_raw.loc[(df_raw['Session Type'] == 
                            'Contributed Talk Presentations') & ab_idx, columns_to_keep]
        df_pstr = df_raw.loc[(df_raw['Session Type'] == 
                            'Contributed Poster Presentations') & ab_idx, columns_to_keep]
        
        # Reindex the DataFrames
        df_con.reset_index(drop=True, inplace=True)
        df_pstr.reset_index(drop=True, inplace=True)

        # Save each DataFrame to a CSV file
        df_con.to_csv(os.path.join(div_dir, 'talks.csv'), index=False)
        df_raw.loc[(df_raw['Session Type'] == 'Contributed Talk Presentations') & ab_idx, 'exported_to_csv'] = 1
        df_pstr.to_csv(os.path.join(div_dir, 'posters.csv'), index=False)  
        df_raw.loc[(df_raw['Session Type'] == 'Contributed Poster Presentations') & ab_idx, 'exported_to_csv'] = 1

        # Update saved_abs_counter
        saved_abs_counter += df_con.shape[0]
        saved_abs_counter += df_pstr.shape[0]

        # Copy the full abstracts file to the division directory
        df_raw.loc[ab_idx, columns_to_keep].to_csv(os.path.join(div_dir, ab_file[:-5] + '.csv'), index=False)

        # Report all files that were saved
        print(f"    {division}: saved abstracts_revised.csv talks.csv, posters.csv.")
    
    # Check if the total number of saved abstracts matches the original count
    if saved_abs_counter != num_abs:
        raise ValueError("Mismatch in the number of abstracts: expected {}, but saved {}".format(num_abs, saved_abs_counter))

    print("All abstracts successfully distributed to CSV files.")

    # Save updated df_raw file (to troubleshoot abstracts not saved to CSV files)
    # output_path = os.path.join(data_root, ab_file[:-5] + '_after_distribution.xlsx')
    # df_raw.to_excel(output_path, index=False)

def get_keywords(data_root, division):

    # Load the keywords file
    keywords_file_path = os.path.join(data_root, 'keywords.xlsx')
    df_key = pd.read_excel(keywords_file_path)

    # Extract and format major group keywords
    major_group_keywords = df_key.loc[df_key['category'].str.lower() == 'major_groups', 'keywords'].iloc[0].split(', ')
    major_group_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in major_group_keywords]

    # Extract and format plant group keywords
    plant_groups_keywords = df_key.loc[df_key['category'].str.lower() == 'plant_groups', 'keywords'].iloc[0].split(', ')
    plant_groups_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in plant_groups_keywords]

    # Extract and format plant group keywords
    animal_groups_keywords = df_key.loc[df_key['category'].str.lower() == 'animal_groups', 'keywords'].iloc[0].split(', ')
    animal_groups_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in animal_groups_keywords]

    # Extract and format plant group keywords
    biology_keywords = df_key.loc[df_key['category'].str.lower() == 'biology', 'keywords'].iloc[0].split(', ')
    biology_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in biology_keywords]
    
    # Check if the division-specific row exists in df_key
    division_row = df_key.loc[df_key['category'].str.lower() == division.lower(), 'keywords']
    
    if not division_row.empty:
        # Extract and format division-specific keywords
        division_keywords = division_row.iloc[0].split(', ')
        division_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in division_keywords]
        
        # # Add columns for division-specific keywords
        # for word in division_keywords:
        #     if word not in df_div.columns:
        #         df_div[word] = None
    else:
        raise ValueError(f"  Division '{division}' not found in keywords file.")

    # Add plant group keywords for DOB
    if division == 'dob':
         major_group_keywords = major_group_keywords + plant_groups_keywords

    # Otherwise, add animal group keywords (can modify for divisions that want both plants and animals)
    if division != 'dob':
        major_group_keywords = major_group_keywords + animal_groups_keywords

    # Make list that combines major_group_keywords and biology_keywords
    major_group_keywords = major_group_keywords + biology_keywords

    # Add division-specific keywords to major_group_keywords
    major_group_keywords = major_group_keywords + division_keywords

    # Add plant group keywords for DOB
    # if division == 'dob':
    #     # Add columns to df_div for each word in plant_groups_keywords
    #     for word in plant_groups_keywords:
    #         df_div[word] = None

    # # Otherwise, add animal group keywords
    # else:
    #     # Add columns to df_div for each word in animal_groups_keywords
    #     for word in animal_groups_keywords:
    #         df_div[word] = None

    # # Add columns to df_div for each word in animal_groups_keywords
    # for word in biology_keywords:
    #     df_div[word] = None

    return major_group_keywords
    ttt=1


def setup_ratings(data_root):
    """
    Create dataframes with columns for rating keywords based on abstracts.

    This function processes abstracts for different divisions, adding columns for specific keywords. 
    It checks for major group keywords and division-specific keywords from a provided keywords file, 
    then adds these as new columns to the abstract dataframes.

    Parameters:
    - inter_data_dir (str): The directory path where intermediate data files are stored.
    - keywords_file_path (str): The file path to the Excel file containing keywords.
    
    Returns:
    - None: The function saves the processed dataframes as CSV files.
    """

    print(' ')
    print("Setting up divisional ratings files ...")

    # List of csv files to load
    csv_files = ['talks', 'posters']

    # Loop through csv_files, load file
    for curr_csv in csv_files:

        # Loop through each division
        for division in divisions_list:
            
            # Directory to save output files
            if division == 'dcb':  
                division = 'dcb_dvm' 
            elif division == 'dvm':
                continue            
            
            # Get keywords for the current division
            keywords = get_keywords(data_root, division)

            # Current divisional paths
            in_path = os.path.join(data_root, div_dir_name, division, curr_csv + '.csv')
            out_path = os.path.join(data_root, div_dir_name, division, curr_csv + '_ratings.csv')

             # Load the abstracts csv file
            df_curr = pd.read_csv(in_path)

            # Create a dataframe that copies 'id', 'clean_title', and 'clean_abstract' for the current division
            df_div = df_curr[df_curr['ab_division'] == division][['id', 'clean_title', 'clean_abstract']].copy()

            # Add column for 'summary'
            df_div['summary'] = None
            
            # Add columns to df_div for each word in major_group_keywords
            for word in keywords:
                df_div[word] = None
            
            # Check if the file already exists
            if not os.path.exists(out_path):
                # Save df_div to a csv file in a subdirectory for each division
                df_div.to_csv(out_path, index=False)
                print(f"  {division} : Saved {curr_csv}_ratings.csv")
            else:
                print(f"  {division} : {curr_csv}_ratings.csv already exists and will not be overwritten.")

    print("Ratings setup complete.")


def setup_weights(data_root):
    """
    Create dataframes with columns for weighting keywords based on abstracts.

    This function processes abstracts for different divisions, adding columns for specific keywords.
    It checks for major group keywords and division-specific keywords from a provided keywords file,
    then adds these as new columns to the abstract dataframes.

    Parameters:
    - data_root (str): The directory path where data files are stored.

    Returns:
    - None: The function saves the processed dataframes as CSV files.
    """

    print(' ')
    print("Setting up divisional weights files ...")

    # Load the keywords file
    keywords_file_path = os.path.join(data_root, 'keywords.xlsx')
    df_key = pd.read_excel(keywords_file_path)

    # Loop through each division
    for division in divisions_list:
        
        # Directory to save output files
        if division == 'dcb':  
            division = 'dcb_dvm' 
        elif division == 'dvm':
            continue      

        # Get keywords for the current division
        keywords = get_keywords(data_root, division)      

        # Current divisional paths
        out_path = os.path.join(data_root, div_dir_name, division, 'keyword_weights.csv')

        # Create a dataframe with the columns 'keyword', 'weight_clustering'
        df_div = pd.DataFrame({'keyword': keywords, 'weight_clustering': [1] * len(keywords),
                              'weight_sequencing': [1] * len(keywords)})

        # Add keyword list
        # df_div = pd.concat([df_div, keywords], ignore_index=True)

        # # Check if the division-specific row exists in df_key
        # division_row = df_key.loc[df_key['category'].str.lower() == division.lower(), 'keywords']
        
        # if not division_row.empty:
        #     # Extract and format division-specific keywords
        #     division_keywords = division_row.iloc[0].split(', ')
        #     division_keywords = [word.strip().replace(' ', '_').replace('-', '_').lower() for word in division_keywords]

        #     # Create a DataFrame for new keywords
        #     new_keywords_df = pd.DataFrame({'keyword': division_keywords, 'weight': [1] * len(division_keywords)})

        #     # Concatenate the new DataFrame with df_div
        #     df_div = pd.concat([df_div, new_keywords_df], ignore_index=True)
            
        # else:
        #     raise ValueError(f"  Division '{division}' not found in keywords file.")

        out_path = os.path.join(data_root, div_dir_name, division, 'keyword_weights.xlsx')

        # Check if the file already exists
        if not os.path.exists(out_path):
            # Save df_div to a csv file in a subdirectory for each division
            df_div.to_excel(out_path, index=False)
            print(f"  {division}: Saved {out_path}")
        else:
            print(f"  {division}: {out_path} already exists and will not be overwritten.")

    print("Weights setup complete.")


def create_keyword_xlsx(inter_data_dir):

    csv_file_path = os.path.join(inter_data_dir, 'contributed_ratings')

    #  Get listing of all csv files in the current directory
    csv_files = [file for file in os.listdir(csv_file_path) if file.endswith('.csv')]

    # Loop through each csv file
    for csv_file in csv_files:

        # Load the CSV file
        df = pd.read_csv(os.path.join(csv_file_path, csv_file))

        # Columns to ignore
        ignore_columns = {'id', 'clean_title', 'clean_abstract', 'summary'}

        # Extract keywords (column names) except the ignored ones
        keywords = [col for col in df.columns if col not in ignore_columns]

        # Create a new DataFrame with 'keyword' and 'weighting'
        keyword_df = pd.DataFrame({'keyword': keywords, 'weight': [1] * len(keywords)})

        # write to Excel file
        keyword_df.to_excel(os.path.join(csv_file_path, csv_file[:-4] + '_keywords.xlsx'), index=False)
        ttt=3
