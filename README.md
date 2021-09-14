# PeopleSun WP4 geospatial tool

TEST 

This geospatial computer tool is part of the PeopleSun project (peoplesun.org).
The project contains a webapp that wraps tools to optimize off-grid systems based on a webmap.

The features of the tool are the following:

- Identification of the buildings from OSM (overpass) API.
- Optimization of network layout of mini-grids with a hub structure. The optimization is based on the Python sgdotlite package.
- Identification of buildings that are located far enough to the rest of the buildings so that it makes sense to provide them with idividual solar-home systems.


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
