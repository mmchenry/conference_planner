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