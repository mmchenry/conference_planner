## Cloning GitHub archive to Google Colab

Instead of downloading the code, one may link Google Colab to the repository. 
This is more onerous to set up, but then allows one to more easily update Colab when software updates become available.
Here are the steps:

- Login to [Colab](https://colab.research.google.com) and create a new notebook. This may appear as a button on a dialog box when you navigate to the Colab site, you or can do this by selecting "New notebook" under the file menu within the notebook.

- Your new notebook will launch with a code cell selected. Paste the following lines into this cell:
    > from google.colab import drive \
    import os \
    drive.mount('/content/drive')
    
    and click on the right-pointing arrow on the left side of the cell to execute the cell.
    When prompted, select "Connect to Google Drive" and then grant permission to allow Colab to access to Drive.

- Click on the "+ Code" button on the top-left of the notebook to create a new cell and then paste and execute the following (Note that you may need to adjust the path address if your Colab notebooks are not in the default location on Drive):
    > os.chdir('/content/drive/MyDrive/Colab Notebooks/') \
    ! git clone https://github.com/mmchenry/conference_planner.git

    If the clone worked, then you should get a response that ends like this:
    > Receiving objects: 100% (7/7), 8.21 KiB | 1.64 MiB/s, done.

- Pulling updates of the archive may be achieved by creating a cell to execute the following command:
    > ! git pull https://github.com/mmchenry/conference_planner.git

    Note that you will again have to update the paths to match the folder addresses on your Google Drive.