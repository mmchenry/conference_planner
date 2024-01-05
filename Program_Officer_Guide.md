The SICB PO needs to initiate the creation of the meeting program with the following steps.

# Download abstract data from X-CD

- Browse to the X-CD web interface for the conference.
- Choose "Submissions overview" (left menu)
- Click on blue "Submissions" button, which will send you to a "Manage Submissions" page.
- Click on the red "Download" button.
- Click on all three "Chk" buttons to select all fields for download
- Scroll down and click on "Download Report" to download an xlsx file (starts with the word "abstracts").

# Setting up Google Drive

- Navigate to your "Colabs Notebooks" folder on [Google Drive](https://drive.google.com).
- Double-clicking on "meeting_planner.ipynb" (within the "conference_planner" folder) should launch that notebook in Google Colab.
- In the first code cell of meeting_planner, you will want to set the root paths for your code and data on your Google Drive. 
- Into the data_root folder, you will want to upload the abstracts file from X-CD and to specify the name of that file (without its extension) in the first cell of meeting_planner.
- You will additionally want to upload [keywords.xlsx](https://github.com/mmchenry/conference_planner/blob/main/keywords.xlsx) to data_root. keywords.xlsx is a spreadsheet that lists the keywords that will be used by GPT-4 to rate the abstracts. A template for this file is included in the present repository and can be downloaded by clicking on this [link](https://github.com/mmchenry/conference_planner/blob/main/keywords.xlsx). 
- Further instructions are offered in the text cells within [meeting_planner](meeting_planner.ipynb).

# Files and directories

The directory structure for Conference Planner data is as follows. Unless noted otherwise, the files and directory are generate by code executed in meeting_planner.ipynb. Note that the code files should be stored separately from the data files, perhaps in a 'Google Colab' directory on Drive. 

- **(root directory)** This directory is accessed only by you, the PO. 
	- **(xlsx file from X-CD)**: You needs to place this file here after download. This will have a filename like 'abstracts_123852.xlsx'.
	- **'abstracts_revised.xlsx'**: This is an edited version of the X-CD file, with duplicates and other erroneous abstracts removed.
	- **'duplicate_primary_contacts.xlsx'**: Listing of abstracts where the primary contact has more than one abstract. These may be kosher, but the idea here is to flag presenters that are possibly signed up for more than one presentation. 
	- **'duplicates.xlsx'**: Listing of abstracts with the same titles. 
	- **'keywords.xlsx'**: Keywords used by GPT-4 to categorize the abstracts. This needs to be copied to the root directory by the PO and a template may be found in the Conference Planner git archive. 
	- **'division_files'** (directory):  Each DPO should have access to their own division directory, with permission granted by the PO.
		- **(directory for each division)**: The files within each divisional directory are explained in the [Divisional Program Officer Guide](Divisional_Program_Officer_guide.md)
			

%% The GPT-4 ratings for each division uses the keywords specific to that division. In addition, the following groups of terms are used by all divisions: major_groups, plant_groups, animal_groups, and biology.  %%