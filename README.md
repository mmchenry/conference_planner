# Conference Planner

The is a set of tools for organizing a conference, developed for the annual meetings of the [Society for Integrative and Comparative Biology](https://sicb.org). 
This can be adapted for other non-profit organizations, but was developed under a [license](LICENSE) that prohibits commercial use.

## Overview 

The meeting organization is administered by the Program Officer (PO) at the level of the society and curated by the divisional program officers (DPOs). 
The code in this repository therefore provides tools for both the PO and DPOs.
To ease deployment, the code and instructions assume use of [Google's Colaboratory](https://colab.research.google.com) platform, which allows for the execution of Python code and uses [Google Drive](https://drive.google.com) to read and write files. 
The Python code may alternatively be run on the local machines of the PO and/or DPOs. 

The present version of the code assumes that the abstracts submitted for the conference will be collected by [X-CD](https://www.x-cd.com), which is a vendor contracted by SICB.
The only reason this matters is that the code assume particular formatting for the data that is collected. 
Any changes to this formatting will require alterations to the code.
Conference Planner starts its work with an XLSX file of the submitted abstracts, provided by X-CD.
The PO's initial work includes combing through the abstracts for duplicates and otherwise erroneous submissions. 
The PO will then distribute the contributed talks and posters to divisional directories, for which the DPOs should be granted access. 

The PO's work includes running code that used GPT-4 to rate how keywords relate the the text of each abstract.
This step is key to the success or failure of the whole project. 
Prior to using GPT-4, the PO and DPOs might collaborate on the listing of keywords that are appropriate for each division. 
These keywords should be of the type that typically differentiates sessions at the meeting.
GPT-4 will be prompted to rate, on a scale from 0 to 1, how well each keyword relates to the abstract text for each presentation.
This step uses the OpenAI API, which costs money (a total of around $50 for the 2024 program), and therefore would ideally be performed only once for each year's planning.

After the ratings are performed, the DPO's primary work begins.
The formulation of sessions is performed in two rounds, both of which draw from GPT-4's keyword ratings.
First, hierarchical clustering is performed to effectively determine the phylogenetic relationships between talks and to defined the major clades of talks.
Next, the code tries numerous combinations of the presentations to find the most their most compatible grouping and sequencing.
Although this steps are automated, the DPOs can adjust the weighting of keywords to determine which are most important for the creation of sessions.
If the DPOs are unsatisfied with the code's results, then they can always resort to manually composing the sessions.

Once the sessions have been established, then the DPOs will determine the session chairs and room assignments. These steps have yet to be coded in Conference Planner.

## Setting up Google Colab

- Both the POs and DPOs will need to have a [Google Colab account](https://colab.research.google.com), which requires a google account and use of [Google Drive](https://drive.google.com).

- Login to [Colab](https://colab.research.google.com) and create a new notebook. This may appear as a button on a dialog box when you navigate to the Colab site, you or can do this by selecting "New notebook" under the file menu within the notebook.

- Your new notebook will launch with a code cell selected. Paste the following lines into this cell:
    > from google.colab import drive \
    import os \
    drive.mount('/content/drive')
    
    and click on the right-pointing arrow on the left side of the cell to execute the cell.
    When prompted, select "Connect to Google Drive" and then grant permission to allow Colab to access to Drive.

- You will next want a copy of Conference Planner in your Colab account, which is accomplished by cloning from its Github repository. Click on the "+ Code" button on the top-left of the notebook to create a new cell and then paste and execute the following (Note that you may need to adjust the path address if your Colab notebooks are not in the default location on Drive):
    > os.chdir('/content/drive/MyDrive/Colab Notebooks/') \
    ! git clone https://github.com/mmchenry/conference_planner.git

    If the clone worked, then you should get a response that ends like this:
    > Receiving objects: 100% (7/7), 8.21 KiB | 1.64 MiB/s, done.

- You now have access to the Python code needed for Conference Planner. 
However, the data will be stored elsewhere on your Drive. 
If you are a DPO, then you will first need the PO to grant you access to your divisional directory.

- At the top of the notebook, rename it to something like "Clone conference planner.ipynb" and then select "Save" in the notebook's File menu. 
This will save a copy of the notebook that you created into the default Colab directory. 
You will want to keep this notebook to pull any new version of the code, if a new version of the repository becomes available.
In the event that you do need to pull a new copy of the code, then you'll have to manually delete your copy of "conference_planner" from Drive before rerunning the notebook.

- Select "Locate on Drive" in the notebook's File menu, to see where the present notebook is located. 
You should see the "conference_planner" folder in the same location, which has the code needed for this project.

## Next steps for the Program Officer

The SICB PO needs to initiate the creation of the meeting program with the following steps.

### Download abstract data from X-CD

- Browse to the X-CD web interface for the conference.
- Choose "Submissions overview" (left menu)
- Click on blue "Submissions" button, which will send you to a "Manage Submissions" page.
- Click on the red "Download" button.
- Click on all three "Chk" buttons to select all fields for download
- Scroll down and click on "Download Report" to download an xlsx file (starts with the word "abstracts").

### Setting up Google Drive

- Navigate to your "Colabs Notebooks" folder on [Google Drive](https://drive.google.com).
- Double-clicking on "meeting_planner.ipynb" (within the "conference_planner" folder) should launch that notebook in Google Colab.
- In the first code cell of meeting_planner, you will want to set the root paths for your code and data on your Google Drive. 
- Into the data_root folder, you will want to upload the abstracts file from X-CD and to specify the name of that file (without its extension) in the first cell of meeting_planner.
- You will additionally want to upload [keywords.xlsx](https://github.com/mmchenry/conference_planner/blob/main/keywords.xlsx) to data_root. 
keywords.xlsx is a spreadsheet that lists the keywords that will be used by GPT-4 to rate the abstracts. 
A template for this file is included in the present repository and can be downloaded by clicking on this [link](https://github.com/mmchenry/conference_planner/blob/main/keywords.xlsx). 
- Further instructions are offered in the text cells within [meeting_planner](meeting_planner.ipynb).

## Next steps for the Divisional Program Officers

- After setting up Google Colab with the Conference Planner code, you will need a directory on your Google Drive to work with the abstracts for your division. This directory should be share with you from the Program Officer.

- You need to have edit permissions to your division folder (please do not rename this folder). 
The folder for your division should have the following files: 

    - **talks.csv**: Simplified listing of the abstract and title on contributed talks
    - **posters.csv**: Similar listing for contributed posters
    - **talks_ratings.csv**: Values for the keyword ratings for talks.
    - **posters_ratings.csv**: Keyword ratings for the posters.
    - **abstracts_revised.csv**: Copy of all abstracts for the meeting.

- Launch [session_planner.ipynb](session_planner.ipynb) from the conference_planner folder in the "Colab Notebooks" folder on Drive.





<!-- 1. [Group abstracts into sessions of 6-8 talks](docs/session_making.md).
1. [Scheduling:Assign each session to a date, time, and room.](docs/scheduling.md) -->
<!-- 1. [Upload resulting schedule to the X-CD database.](docs/download_upload.md). -->

<!-- ## Directory structure

The root path should include the following directories. Within each, there is a subdirectory for the year of the meeting to be organized:

- **source_data:** Has the downloaded abstracts xlsx file (e.g., "abstracts_123852.xlsx") and the list of keywords for each division ("keywords.xls").
- **intermediate_data:** Location for needed data files generated as part of the processing of the data.
- **output_data:** Files needed to compose the program saved here.


## Operating the code -->




