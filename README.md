# Description

This open-source computer tool is part of the PeopleSuN project (https://www.peoplesun.org), which contains a web-based app that aims to optimize the network structure as well as the power supply system for mini-grid systems.

The features of the tool are listed below:

- Automatic identification of buildings from the OpenStreetMap inside a given boundary.
- Network optimization of mini-grids based on the Python package /*sgdotlite*/.
- Optimization of the power supply systems for mini-grids (PV, battery, and diesel) using a Python-based tool called /*Offgridders*/. 
- Automatic identification of buildings that are better to be served by idividual solar home systems.


## Get started

### Installation

1. From the root of the repository, first create a virtual environment (here called venv) using the following command:
   python3 -m venv venv

Note that conda can also be used to set up a virtual environment

2. Activate the virtual environment running the following command:

i. On Linux, MacOs
source venv/bin/activate

ii. On Windows
venv\Scripts\bin\activate

3. Install the required packages using the following command:
   pip install -r requirements.txt

### Launching of the app

1. Run the following command to start the FastAPI server:
   uvicorn fastapi_app.main:app --reload

2. Open the following URL in a browser (preferably on Chrome or Firefox):
   http://127.0.0.1:8000/
