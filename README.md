# Conference Planner

The is a set of tools for organizing a conference. 
It was developed in particular for the annual meetings of the [Society for Integrative and Comparative Biology](https://sicb.org). 
This can be adapted to other non-profit organizations, but its [license](LICENSE) prohibits commercial use.

## Overview of workflow

This code would aim to achieve the following steps, which are currently executed by the divisional program officers of SICB.

1. [Download abstract data from X-CD](\docs\download_upload.md).
1. [Group abstracts into sessions of 6-8 talks](\docs\session_making.md).
1. [Scheduling:Assign each session to a date, time, and room.](\docs\scheduling.md)
1. [Upload resulting schedule to the X-CD database.](\docs\download_upload.md).

## Directory structure

The root path should include the following directories. Within each, there is a subdirectory for the year of the meeting to be organized:

- **source_data:** Has the downloaded abstracts xlsx file (e.g., "abstracts_123852.xlsx") and the list of keywords for each division ("keywords.xls").
- **intermediate_data:** Location for needed data files generated as part of the processing of the data.
- **output_data:** Files needed to compose the program saved here.


## Operating the code




