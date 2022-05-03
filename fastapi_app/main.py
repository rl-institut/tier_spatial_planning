from sqlalchemy.sql.expression import column, false, true
from sqlalchemy.sql.sqltypes import Boolean
import fastapi_app.tools.boundary_identification as bi
import fastapi_app.tools.coordinates_conversion as conv
import fastapi_app.tools.shs_identification as shs_ident
import fastapi_app.tools.io as io
import fastapi_app.models as models
from fastapi.param_functions import Query
from fastapi import FastAPI, Request, Depends, BackgroundTasks, File, UploadFile
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_app.database import SessionLocal, engine
from sqlalchemy.orm import Session, raiseload
import sqlite3
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.grid_optimizer import GridOptimizer
import math
import urllib.request
import ssl
import json
import pandas as pd
import numpy as np
import time
import os
import aiofiles
# for debugging
import uvicorn
# for appending to the dictionary
from collections import defaultdict
# for sending an array of data from JS to the fastAPI
from typing import Any, Dict, List, Union

app = FastAPI()

app.mount("/fastapi_app/static",
          StaticFiles(directory="fastapi_app/static"), name="static")

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="fastapi_app/templates")

# define different directories for:
# (1) database: *.csv files for nodes and links,
# (2) inputs: input excel files (cost data and timeseries) for offgridders + web app import and export files, and
# (3) outputs: offgridders results
directory_parent = "fastapi_app"

directory_database = os.path.join(directory_parent, 'data', 'database').replace("\\", "/")
full_path_nodes = os.path.join(directory_database, 'nodes.csv').replace("\\", "/")
full_path_links = os.path.join(directory_database, 'links.csv').replace("\\", "/")
os.makedirs(directory_database, exist_ok=True)

directory_inputs = os.path.join(directory_parent, 'data', 'inputs').replace("\\", "/")
full_path_import_export = os.path.join(directory_inputs, 'import_export.xlsx').replace("\\", "/")
full_path_offgridders_inputs = os.path.join(directory_inputs, 'input_data.xlsx').replace("\\", "/")
full_path_offgridders_timeseries = os.path.join(
    directory_inputs, 'site_data.xlsx').replace("\\", "/")
os.makedirs(directory_inputs, exist_ok=True)

directory_outputs = os.path.join(directory_parent, 'data', 'outputs').replace("\\", "/")
os.makedirs(directory_outputs, exist_ok=True)

# this is to avoid problems in "urllib" by not authenticating SSL certificate, otherwise following error occurs:
# urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)>
ssl._create_default_https_context = ssl._create_unverified_context

# define the template for importing json data in the form of arrays from js to python
json_object = Dict[Any, Any]
json_array = List[Any]
import_structure = Union[json_array, json_object]


# --------------------- REDIRECT REQUEST TO FAVICON LOG ----------------------#


@app.get("/favicon.ico")
async def redirect():
    """ Redirects request to location of favicon.ico logo in static folder """
    response = RedirectResponse(url='/fastapi_app/static/favicon.ico')
    return response


# ************************************************************/
# *                     IMPORT / EXPORT                      */
# ************************************************************/

@app.post("/export_data/")
async def export_data(
        generate_export_file_request: models.GenerateExportFileRequest):
    """
    Generates an Excel file from the database tables (*.csv files) and the
    webapp settings. The file is stored in fastapi_app/import_export/temp.xlsx

    Parameters
    ----------
    generate_export_file_request (fastapi_app.models.GenerateExportFileRequest):
        Basemodel request object containing the data send to the request as attributes.
    """

    # read nodes and links from *.csv files
    # then convert their type from dictionary to data frame
    nodes = await database_read(nodes_or_links='nodes')
    links = await database_read(nodes_or_links='links')
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)

    # get all settings defined in the web app
    settings = [element for element in generate_export_file_request]
    settings_df = pd.DataFrame({"Setting": [x[0] for x in settings],
                                "value": [x[1] for x in settings]}).set_index('Setting')

    # create the *.xlsx file with sheets for nodes, links and settings
    with pd.ExcelWriter(full_path_import_export) as writer:  # pylint: disable=abstract-class-instantiated
        nodes_df.to_excel(excel_writer=writer, sheet_name='nodes',
                          header=nodes_df.columns, index=False)
        links_df.to_excel(excel_writer=writer, sheet_name='links',
                          header=links_df.columns, index=False)
        settings_df.to_excel(excel_writer=writer, sheet_name='settings')

    # TO DO: formatting of the excel file


@app.get("/download_export_file",
         responses={200: {"description": "xlsx file containing the information about the configuration.",
                          "content": {"static/io/test_excel_node.xlsx": {"example": "No example available."}}}})
async def download_export_file():
    file_name = 'temp.xlsx'
    # Download xlsx file
    file_path = os.path.join(directory_parent, f"import_export/{file_name}")

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="backup.xlsx")
    else:
        return {"error": "File not found!"}


@app.post("/import_data")
async def import_data(import_files: import_structure = None):

    # empty *.csv files cotaining nodes and links
    await database_initialization(nodes=True, links=True)

    # add nodes from the 'nodes' sheet of the excel file to the 'nodes.csv' file
    # TODO: update the template for adding nodes
    nodes = import_files['nodes_to_import']
    links = import_files['links_to_import']
    if len(nodes) > 0:
        database_add(add_nodes=True, add_links=False, inlet=nodes)

    if len(links) > 0:
        database_add(add_nodes=False, add_links=True, inlet=links)

    # ------------------------------ HANDLE REQUEST ------------------------------#


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("home.html", {
        "request": request
    })


@app.get("/database_initialization/{nodes}/{links}")
async def database_initialization(nodes, links):
    # creating the csv files
    # - in case these files do not exist they will be created here
    # - each time the code runs from the beginning, the old csv files will be replaced with new blank ones
    header_nodes = [
        "latitude",
        "longitude",
        "node_type",
        "consumer_type",
        "consumer_detail",
        "average_consumption",
        "peak_demand",
        "is_connected",
        "how_added"
    ]
    header_links = [
        "lat_from",
        "lon_from",
        "lat_to",
        "lon_to",
        "link_type",
        "length"
    ]
    if nodes:
        pd.DataFrame(columns=header_nodes).to_csv(full_path_nodes, index=False)

    if links:
        pd.DataFrame(columns=header_links).to_csv(full_path_links, index=False)


# add new manually-selected nodes to the *.csv file
# TODO: update the template for adding nodes
@app.post("/database_add_manual")
async def database_add_manual(
        add_node_request: models.AddNodeRequest):

    headers = pd.read_csv(full_path_nodes).columns
    nodes = {}
    nodes[headers[0]] = [add_node_request.latitude]
    nodes[headers[1]] = [add_node_request.longitude]
    nodes[headers[3]] = [add_node_request.node_type]
    nodes[headers[4]] = [add_node_request.consumer_type]
    nodes[headers[5]] = [add_node_request.consumer_detail]
    nodes[headers[5]] = [add_node_request.average_consumption]
    nodes[headers[5]] = [add_node_request.peak_demand]
    nodes[headers[7]] = [add_node_request.is_connected]
    nodes[headers[8]] = [add_node_request.how_added]

    database_add(add_nodes=True, add_links=False, inlet=nodes)


# add new nodes/links to the database
def database_add(add_nodes: bool,
                 add_links: bool,
                 inlet: dict):

    # updating csv files based on the added nodes
    if add_nodes:
        nodes = inlet
        # defining the precision of data
        df = pd.DataFrame.from_dict(nodes)
        df.latitude = df.latitude.map(lambda x: "%.6f" % x)
        df.longitude = df.longitude.map(lambda x: "%.6f" % x)

        # getting existing latitudes from the csv file as a list of float numbers
        # and checking if some of the new nodes already exist in the database or not
        # and then excluding the entire row from the dataframe that is going to be added to the csv file
        df_existing = list(pd.read_csv(full_path_nodes)["latitude"])
        for latitude in [float(x) for x in list(df["latitude"])]:
            if latitude in df_existing:
                df = df[df.latitude != str(latitude)]

        # finally adding the refined dataframe (if it is not empty) to the existing csv file
        if len(df.index) != 0:
            df.to_csv(full_path_nodes, mode='a', header=False, index=False, float_format='%.0f')

    if add_links:
        links = inlet
        # defining the precision of data
        df = pd.DataFrame.from_dict(links)
        df.lat_from = df.lat_from.map(lambda x: "%.6f" % x)
        df.lon_from = df.lon_from.map(lambda x: "%.6f" % x)
        df.lat_to = df.lat_to.map(lambda x: "%.6f" % x)
        df.lon_to = df.lon_to.map(lambda x: "%.6f" % x)

        # adding the links to the existing csv file
        if len(df.index) != 0:
            df.to_csv(full_path_links, mode='a', header=False, index=False, float_format='%.0f')


# remove some nodes from the database
def database_remove_nodes(nodes,
                          nodes_index_removing):

    for index in nodes.index:
        if index in nodes_index_removing:
            nodes.drop(labels=index, axis=0, inplace=True)

    # storing the nodes in the database (updating the existing CSV file)
    nodes = nodes.reset_index(drop=True)
    database_add(add_nodes=True, add_links=False, inlet=nodes.to_dict())


@app.get("/database_to_js/{nodes_or_links}")
async def database_read(nodes_or_links: str):

    # importing nodes and links from the csv files to the map
    if nodes_or_links == 'nodes':
        nodes_list = json.loads(pd.read_csv(full_path_nodes).to_json())
        return nodes_list
    else:
        links_list = json.loads(pd.read_csv(full_path_links).to_json())
        return links_list


@app.post("/database_add_remove_automatic/{add_remove}")
async def database_add_remove_automatic(
        add_remove: str,
        selectBoundariesRequest: models.SelectBoundariesRequest):

    boundary_coordinates = selectBoundariesRequest.boundary_coordinates

    # latitudes and longitudes of all buildings in the selected boundary
    latitudes = [x[0] for x in boundary_coordinates]
    longitudes = [x[1] for x in boundary_coordinates]

    if add_remove == "add":
        # min and max of latitudes and longitudes are sent to the overpass to get
        # a large rectangle including (maybe) more buildings than selected
        min_latitude = min(latitudes)
        min_longitude = min(longitudes)
        max_latitude = max(latitudes)
        max_longitude = max(longitudes)
        url = f'https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500][bbox:{min_latitude},{min_longitude},{max_latitude},{max_longitude}];(way["building"="yes"];relation["building"];);out body;>;out skel qt;'
        url_formated = url.replace(" ", "+")
        with urllib.request.urlopen(url_formated) as url:
            data = json.loads(url.read().decode())

        # first converting the json file, which is delievered by overpass to geojson,
        # then obtaining coordinates and surface areas of all buildings inside the
        # 'big' rectangle.
        formated_geojson = bi.convert_overpass_json_to_geojson(data)
        building_coord = bi.obtain_mean_coordinates_from_geojson(formated_geojson)

        # excluding the buildings which are outside the drawn boundary
        features = formated_geojson['features']
        mask_building_within_boundaries = {
            key: bi.is_point_in_boundaries(
                value,
                boundary_coordinates) for key, value in building_coord.items()}
        filtered_features = [feature for feature in features
                             if mask_building_within_boundaries[
                                 feature['property']['@id']]
                             ]
        formated_geojson['features'] = filtered_features
        building_coordidates_within_boundaries = {
            key: value for key, value in building_coord.items()
            if mask_building_within_boundaries[key]
        }

        # creating a dictionary from the given nodes and sending this dictionary
        # to the 'database_add' function to store nodes properties in the database
        nodes = defaultdict(list)
        for label, coordinates in building_coordidates_within_boundaries.items():
            nodes["latitude"].append(coordinates[0])
            nodes["longitude"].append(coordinates[1])
            nodes["node_type"].append("consumer")
            nodes["consumer_type"].append("household")
            nodes["consumer_detail"].append("default")

            # average consumption and peak demand are not yet known
            nodes["average_consumption"].append('')
            nodes["peak_demand"].append('')

            # it is assumed that all nodes are parts of the mini-grid
            # later, when the shs candidates are obtained, the corresponding
            # values will be changed to 'False'
            nodes["is_connected"].append(True)

            # the node is selected automatically after drawing boundaries
            nodes["how_added"].append("automatic")

        # storing the nodes in the database
        database_add(add_nodes=True, add_links=False, inlet=nodes)

    else:
        # reading the existing CSV file of nodes, and then removing the corresponding row
        df = pd.read_csv(full_path_nodes)
        number_of_nodes = df.shape[0]
        for index in range(number_of_nodes):
            if bi.is_point_in_boundaries(point_coordinates=(df.to_dict()['latitude'][index], df.to_dict()['longitude'][index]), boundaries=boundary_coordinates):
                df.drop(labels=index, axis=0, inplace=True)

        # removing all nodes and links
        await database_initialization(nodes=True, links=True)

        # storing the nodes in the database (updating the existing CSV file)
        df = df.reset_index(drop=True)
        database_add(add_nodes=True, add_links=False, inlet=df.to_dict())


@ app.post("/optimize_grid/")
async def optimize_grid(optimize_grid_request: models.OptimizeGridRequest,
                        background_tasks: BackgroundTasks):

    # create GridOptimizer object
    opt = GridOptimizer()

    # get nodes from the database (CSV file) as a dictionary
    # then convert it again to a panda dataframe for simplicity
    # TODO: check the format of nodes from the database_read()
    nodes = await database_read(nodes_or_links='nodes')
    nodes = pd.DataFrame.from_dict(nodes)

    # if there is no element in the nodes, optimization will be terminated
    if len(nodes) == 0:
        return {
            "code": "success",
            "message": "Empty grid cannot be optimized!"
        }

    # initialite the database (remove contents of the CSV files)
    # otherwise, when clicking on the 'optimize' button, the existing system won't be removed
    await database_initialization(nodes=True, links=True)

    # nodes obtained from a previous optimization (e.g., poles)
    # will not be considered in the grid optimization
    nodes_index_removing = []
    for node_index in nodes.index:
        if ('optimization' in nodes.how_added[node_index]):
            nodes_index_removing.append(node_index)

    database_remove_nodes(nodes=nodes,
                          nodes_index_removing=nodes_index_removing)

    # create a new "grid" object from the Grid class
    grid = Grid(
        capex_pole=optimize_grid_request.cost_pole,
        capex_connection=optimize_grid_request.cost_connection,
        capex_interpole=optimize_grid_request.cost_interpole_cable,
        capex_distribution=optimize_grid_request.cost_distribution_cable,
        pole_max_connection=optimize_grid_request.max_connection_poles
    )

    # make sure that the new grid object is empty before adding nodes to it
    grid.clear_nodes()
    grid.clear_links()

    # exclude solar-home-systems and poles from the grid optimization
    for node_index in nodes.index:
        if (nodes.is_connected[node_index]) and (not nodes.node_type[node_index] == 'pole'):

            # add all consumers which are not served by solar-home-systems
            grid.add_node(
                label=str(node_index),
                longitude=nodes.longitude[node_index],
                latitude=nodes.latitude[node_index],
                node_type=nodes.node_type[node_index],
                is_connected=nodes.is_connected[node_index]
            )

    # convert all (long,lat) coordinates to (x,y) coordinates and update
    # the Grid object, which is necessary for the GridOptimizer
    grid.convert_lonlat_xy()

    # in case the grid contains 'poles' from the previous optimization
    # they must be removed, becasue the grid_optimizer will calculate
    # new locations for poles considering the newly added nodes
    grid.clear_poles()

    # calculate the minimum number of poles based on the
    # maximum number of connectins at each pole
    if grid.pole_max_connection == 0:
        min_number_of_poles = 1
    else:
        min_number_of_poles = (
            int(np.ceil(grid.nodes.shape[0]/(grid.pole_max_connection)))
        )

    # obtain the optimal number of poles by increasing the minimum number of poles
    # and each time applying the kmeans clustering algorithm and minimum spanning tree
    number_of_poles = opt.find_opt_number_of_poles(
        grid=grid,
        min_n_clusters=min_number_of_poles
    )

    number_of_relaxation_steps_nr = optimize_grid_request.number_of_relaxation_steps_nr

    opt.nr_optimization(grid=grid,
                        number_of_poles=number_of_poles,
                        number_of_relaxation_steps=number_of_relaxation_steps_nr,
                        first_guess_strategy='random',
                        save_output=False,
                        number_of_hill_climbers_runs=0)

    # get all poles obtained by the network relaxation method
    poles = grid.poles().reset_index(drop=True)

    # remove the unnecessary columns to make it compatible with the CSV files
    # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
    poles.drop(labels=['x', 'y', 'cluster_label', 'segment', 'type_fixed',
                       'allocation_capacity'], axis=1, inplace=True)

    # store the list of poles in the "node" database
    database_add(add_nodes=True, add_links=False, inlet=poles.to_dict())

    # get all links obtained by the network relaxation method
    links = grid.links.reset_index(drop=True)

    # remove the unnecessary columns to make it compatible with the CSV files
    # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
    links.drop(labels=['x_from', 'y_from', 'x_to', 'y_to'], axis=1, inplace=True)

    # store the list of poles in the "node" database
    database_add(add_nodes=False, add_links=True, inlet=links.to_dict())


@app.post("/shs_identification/")
def identify_shs(shs_identification_request: models.ShsIdentificationRequest):

    print("starting shs_identification...")

    # res = db.execute("select * from nodes")
    # nodes = res.fetchall()

    if len(nodes) == 0:
        return {
            "code": "success",
            "message": "No nodes in table, no identification to be performed"
        }

    # use latitude of the node that is the most west to set origin of x coordinates
    ref_latitude = min([node[1] for node in nodes])
    # use latitude of the node that is the most south to set origin of y coordinates
    ref_longitude = min([node[2] for node in nodes])

    nodes_df = shs_ident.create_nodes_df()

    cable_price_per_meter = shs_identification_request.cable_price_per_meter_for_shs_mst_identification
    additional_price_for_connection_per_node = shs_identification_request.connection_cost_to_minigrid

    for node in nodes:
        latitude = math.radians(node[1])
        longitude = math.radians(node[2])

        x, y = conv.xy_coordinates_from_latitude_longitude(
            latitude=latitude,
            longitude=longitude,
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude)

        node_label = node[0]
        required_capacity = node[6]
        max_power = node[7]
        # is_connected = node[8]
        if node[4] == "low-demand":
            shs_price = shs_identification_request.price_shs_ld
        elif node[4] == "medium-demand":
            shs_price = shs_identification_request.price_shs_md
        elif node[4] == "high-demand":
            shs_price = shs_identification_request.price_shs_hd

        shs_ident.add_node(nodes_df, node_label, x, y,
                           required_capacity, max_power, shs_price=shs_price)
    links_df = shs_ident.mst_links(nodes_df)
    start_time = time.time()
    if shs_identification_request.algo == "mst1":
        nodes_to_disconnect_from_grid = shs_ident.nodes_to_disconnect_from_grid(
            nodes_df=nodes_df,
            links_df=links_df,
            cable_price_per_meter=cable_price_per_meter,
            additional_price_for_connection_per_node=additional_price_for_connection_per_node)
        print(
            f"execution time for shs identification (mst1): {time.time() - start_time} s")
    else:
        print("issue with version parameter of shs_identification_request")
        return 0

    # sqliteConnection = sqlite3.connect(grid_db)
    # conn = sqlite3.connect(grid_db)
    # cursor = conn.cursor()

    """
    for index in nodes_df.index:
        if index in nodes_to_disconnect_from_grid:
            sql_delete_query = (
                f""UPDATE nodes
                SET node_type = 'shs'
                WHERE  id = {index};
                "")
        else:
            sql_delete_query = (
                f""UPDATE nodes
                SET node_type = 'consumer'
                WHERE  id = {index};
                "")
        cursor.execute(sql_delete_query)
        sqliteConnection.commit()
    cursor.close()

    # commit the changes to db
    conn.commit()
    # close the connection
    conn.close()

    return {
        "code": "success",
        "message": "shs identified"
    }
    """

# -------------------------- FUNCTION FOR DEBUGGING-------------------------- #


def debugging_mode():
    """
    if host="0.0.0.0" and port=8000 does not work, the following can be used:
        host="127.0.0.1", port=8080
    """
    uvicorn.run(app, host="127.0.0.1", port=8080)
