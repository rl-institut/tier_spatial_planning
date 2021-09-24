from re import X
from fastapi.params import Header
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

app = FastAPI()

app.mount("/fastapi_app/static",
          StaticFiles(directory="fastapi_app/static"), name="static")

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="fastapi_app/templates")

grid_db = "grid.db"

path = "fastapi_app"

dir_name = os.path.join(path, "data").replace("\\", "/")
nodes_file = "nodes.csv"
links_file = "links.csv"
full_path_nodes = os.path.join(dir_name, nodes_file).replace("\\", "/")
full_path_nodes = os.path.join(dir_name, nodes_file).replace("\\", "/")
full_path_links = os.path.join(dir_name, links_file).replace("\\", "/")

# ---------------------------- SET UP grid.db DATABASE -----------------------#


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

# --------------------- REDIRECT REQUEST TO FAVICON LOG ----------------------#


@app.get("/favicon.ico")
async def redirect():
    """ Redirects request to location of favicon.ico logo in static folder """
    response = RedirectResponse(url='/fastapi_app/static/favicon.ico')
    return response

# --------------------------- IMPORT/EXPORT FEATURE --------------------------#


@app.post("/generate_export_file/")
async def generate_export_file(
        generate_export_file_request: models.GenerateExportFileRequest,
        db: Session = Depends(get_db)):
    """
    Generates an Excel file from the grid.db database tables and from the
    webapp setting. The file is stored in fastapi_app/import_export/temp.xlsx

    Parameters
    ----------
    generate_export_file_request (fastapi_app.models.GenerateExportFileRequest):
        Basemodel request object containing the data send to the request as attributes.

    db (sqlalchemy.orm.Session):
         Establishes conversation with the {grid_db} database.
    """
    # CREATE NODES DATAFRAME FROM DATABASE
    res_nodes = db.execute("select * from nodes")
    nodes_table = res_nodes.fetchall()

    nodes_df = io.create_empty_nodes_df()

    for node in nodes_table:
        nodes_df.at[node[0]] = node[1:]

    # CREATE LINKS DATAFRAME FROM DATABASE
    res_links = db.execute("select * from links")
    links_table = res_links.fetchall()

    links_df = io.create_empty_links_df()

    for link in links_table:
        links_df.at[link[0]] = link[1:]

    settings = [element for element in generate_export_file_request]

    settings_df = pd.DataFrame({"Setting": [x[0] for x in settings],
                                "value": [x[1] for x in settings]}).set_index('Setting')

    # Create xlsx file with sheets for nodes and for links
    file_name = 'temp.xlsx'
    with pd.ExcelWriter(f'{path}/import_export/{file_name}') as writer:
        nodes_df.to_excel(excel_writer=writer, sheet_name='nodes', header=nodes_df.columns)
        links_df.to_excel(excel_writer=writer, sheet_name='links', header=links_df.columns)
        settings_df.to_excel(excel_writer=writer, sheet_name='settings')


@app.get("/download_export_file",
         responses={200: {"description": "xlsx file containing the information about the configuration.",
                          "content": {"static/io/test_excel_node.xlsx": {"example": "No example available."}}}})
async def download_export_file(db: Session = Depends(get_db)):
    file_name = 'temp.xlsx'
    # Download xlsx file
    file_path = os.path.join(path, f"import_export/{file_name}")

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="backup.xlsx")
    else:
        return {"error": "File not found!"}


@app.post("/import_config")
async def import_config(file: UploadFile = File(...)):

    content_file = await file.read()
    async with aiofiles.open(f"{path}/import_export/import.xlsx", 'wb') as out_file:
        await out_file.write(content_file)

    # Empty Database tables
    clear_nodes_table()
    clear_links_table()

    # Populate nodes table from nodes sheet of file
    nodes_df = pd.read_excel(f"{path}/import_export/import.xlsx",
                             sheet_name="nodes",
                             engine="openpyxl")

    conn = sqlite3.connect(grid_db)
    cursor = conn.cursor()

    records = [(
        str(nodes_df.iloc[i]['label']),
        float(nodes_df.iloc[i]['latitude']),
        float(nodes_df.iloc[i]['longitude']),
        float(nodes_df.iloc[i]['area']),
        str(nodes_df.iloc[i]['node_type']),
        bool(nodes_df.iloc[i]['type_fixed']),
        float(nodes_df.iloc[i]['required_capacity']),
        float(nodes_df.iloc[i]['max_power']),
        float(nodes_df.iloc[i]['is_connected'])
    ) for i in range(nodes_df.shape[0])]

    cursor.executemany(
        'INSERT INTO nodes VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)', records)

    # Populate links table from links sheet of file
    links_df = pd.read_excel(f"{path}/import_export/import.xlsx",
                             sheet_name="links",
                             engine="openpyxl")

    records = [(
        str(links_df.iloc[i]['label']),
        float(links_df.iloc[i]['latitude_from']),
        float(links_df.iloc[i]['longitude_from']),
        float(links_df.iloc[i]['latitude_to']),
        float(links_df.iloc[i]['longitude_to']),
        str(links_df.iloc[i]['type']),
        float(links_df.iloc[i]['distance'])
    ) for i in range(links_df.shape[0])]

    cursor.executemany(
        'INSERT INTO links VALUES(?, ?, ?, ?, ?, ?, ?)', records)

    # commit the changes to db
    conn.commit()
    # close the connection
    conn.close()

    # Collect settings for settings tab and return them as a dict
    settings_df = pd.read_excel(f"{path}/import_export/import.xlsx",
                                sheet_name="settings",
                                engine="openpyxl").set_index('Setting')
    settings = {index: row['value'].item() for index, row in settings_df.iterrows()}

    return settings
    # ------------------------------ HANDLE REQUEST ------------------------------#


@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("home.html", {
        "request": request
    })


# Creating the csv files
# - In case these files do not exist they will be created here
# - Each time the code runs from the beginning, the old csv files will be deleted and new blank ones will be created
@app.get("/csv_files_initialization")
async def csv_files_initialization():
    header_nodes = [
        "latitude",
        "longitude",
        "area",
        "peak_demand",
        "node_type",
        "is_connected",
        "how_added"
    ]
    header_links = [
        "lat_from",
        "long_from",
        "lat_to",
        "long_to",
        "link_type",
        "cable_thickness",
        "length"
    ]
    pd.DataFrame(columns=header_nodes).to_csv(full_path_nodes, index=False)
    pd.DataFrame(columns=header_links).to_csv(full_path_links, index=False)


def db_add(add_nodes: bool,
           add_links: bool,
           nodes: dict):
    if add_nodes:
        df = pd.DataFrame.from_dict(nodes)
        df["latitude"] = df["latitude"].map(lambda x: "%.6f" % x)
        df["longitude"] = df["longitude"].map(lambda x: "%.6f" % x)
        df["area"] = df["area"].map(lambda x: "%.2f" % x)
        df.to_csv(full_path_nodes, mode='a', header=False, index=False, float_format='%.0f')


@app.post("/db_add/{add_nodes}/{add_links}")
async def db_add_from_js(
        add_nodes: bool,
        add_links: bool,
        add_node_request: models.AddNodeRequest):

    if add_nodes:
        headers = pd.read_csv(full_path_nodes).columns
        nodes = {}
        nodes[headers[0]] = [add_node_request.latitude]
        nodes[headers[1]] = [add_node_request.longitude]
        nodes[headers[2]] = [add_node_request.area]
        nodes[headers[3]] = [add_node_request.peak_demand]
        nodes[headers[4]] = [add_node_request.node_type]
        nodes[headers[5]] = [add_node_request.is_connected]
        nodes[headers[6]] = [add_node_request.how_added]

        db_add(add_nodes, add_links, nodes)

    if add_links:
        print("hi")


@app.get("/csv_files_reading/{nodes}/{links}")
async def csv_files_reading(nodes: bool, links: bool, request: Request):
    if nodes:
        nodes_list = json.loads(pd.read_csv(full_path_nodes).to_json())
        return nodes_list
    if links:
        links_list = pd.read_csv(full_path_links)
        return links_list


@app.get("/nodes_db_html")
async def get_nodes(request: Request, db: Session = Depends(get_db)):
    """
    This funtcion access and returns the entries of the nodes table of the grid database

    Parameters
    ----------
    request:
        url get request called to invoke function

    db:
        database to be accessed

    """
    res = db.execute("select * from nodes")
    result = res.fetchall()
    return result


@app.get("/links_db_html")
async def get_links(request: Request, db: Session = Depends(get_db)):
    res = db.execute("select * from links")
    result = res.fetchall()
    return result


@app.post("/select_boundaries/{add_remove}")
async def select_boundaries_add_remove(
        add_remove: str,
        selectBoundariesRequest: models.SelectBoundariesRequest):

    boundary_coordinates = selectBoundariesRequest.boundary_coordinates

    if add_remove == "add":
        # latitudes and longitudes of all buildings in the selected boundary
        latitudes = [x[0] for x in boundary_coordinates]
        longitudes = [x[1] for x in boundary_coordinates]

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
        building_coord, building_area = bi.obtain_areas_and_mean_coordinates_from_geojson(
            formated_geojson)

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
        # to the 'db_add' function to store nodes properties in the database
        nodes = defaultdict(list)
        for label, coordinates in building_coordidates_within_boundaries.items():
            nodes["latitude"].append(coordinates[0])
            nodes["longitude"].append(coordinates[1])
            nodes["area"].append(building_area[label])
            # a very rough estimation for peak_demand at each node
            peak_demand_per_sq_meter = 4
            nodes["peak_demand"].append(building_area[label] * peak_demand_per_sq_meter)
            # categorization of node_type based on the peak_demand value
            if nodes["peak_demand"][-1] >= 100:
                nodes["node_type"].append("high-demand")
            elif 40 < nodes["peak_demand"][-1] < 100:
                nodes["node_type"].append("medium-demand")
            else:
                nodes["node_type"].append("low-demand")
            # it is assumed that all nodes are parts of the mini-grid
            nodes["is_connected"].append(True)

            # the node is selected automatically after drawing boundaries
            nodes["how_added"].append("automatic")

        # storing the nodes in the database
        db_add(add_nodes=True, add_links=False, nodes=nodes)
        # return formated_geojson

    else:
        for node in boundary_coordinates:
            if bi.is_point_in_boundaries(point_coordinates=(node[1], node[2]), boundaries=boundary_coordinates):
                clear_single_node(node[0])

"""
@app.post("/select_boundaries_remove")
async def select_boundaries_remove(
        selectBoundariesRequest: models.SelectBoundariesRequest,
        db: Session = Depends(get_db)):

    boundary_coordinates = selectBoundariesRequest.boundary_coordinates

    res = db.execute("select * from nodes")
    nodes = res.fetchall()

    for node in nodes:
        if bi.is_point_in_boundaries(point_coordinates=(node[1], node[2]), boundaries=boundary_coordinates):
            clear_single_node(node[0])

    return {
        "code": "success",
        "message": "node deleted from db"
    }
"""


@ app.post("/add_node/")
async def add_node(add_node_request: models.AddNodeRequest,
                   background_tasks: BackgroundTasks,
                   db: Session = Depends(get_db)):
    nodes = models.Nodes()

    nodes.latitude = add_node_request.lat
    nodes.longitude = add_node_request.longitude
    nodes.area = add_node_request.area
    nodes.node_type = add_node_request.node_type
    nodes.fixed_type = add_node_request.fixed_type
    nodes.required_capacity = add_node_request.required_capacity
    nodes.max_power = add_node_request.max_power
    nodes.is_connected = add_node_request.is_connected

    db.add(nodes)
    db.commit()

    return {
        "code": "success",
        "message": "node added to db"
    }


@ app.post("/optimize_grid/")
async def optimize_grid(optimize_grid_request: models.OptimizeGridRequest,
                        background_tasks: BackgroundTasks,
                        db: Session = Depends(get_db)):
    # Create GridOptimizer object
    opt = GridOptimizer()

    res = db.execute("select * from nodes")
    nodes = res.fetchall()

    # if nodes db is empty, do not perform optimization
    if len(nodes) == 0:
        return {
            "code": "success",
            "message": "empty grid cannot be optimized"
        }

    for node in nodes:
        node_index = node[0]
        node_type = node[4]
        type_fixed = node[5]

        if (node_type == 'pole') and (not type_fixed):
            clear_single_node(node_index)

    # Create new grid object
    grid = Grid(price_meterhub=optimize_grid_request.price_pole,
                price_household=optimize_grid_request.price_household,
                price_interhub_cable_per_meter=optimize_grid_request.price_pole_cable,
                price_distribution_cable_per_meter=optimize_grid_request.price_distribution_cable,
                default_hub_capacity=optimize_grid_request.max_connection_poles)
    # Make sure that new grid object is empty before adding nodes to it
    grid.clear_nodes_and_links()

    # use latitude of the node that is the most west to set origin of x coordinates
    ref_latitude = math.radians(min([node[1] for node in nodes]))
    # use latitude of the node that is the most south to set origin of y coordinates
    ref_longitude = math.radians(min([node[2] for node in nodes]))
    for node in nodes:
        node_index = node[0]
        latitude = node[1]
        longitude = node[2]
        node_type = node[4]
        type_fixed = bool(node[5])

        if (not node_type == 'shs') and ((not node_type == 'pole') or (node_type == "pole" and type_fixed == True)):

            x, y = conv.xy_coordinates_from_latitude_longitude(
                latitude=latitude,
                longitude=longitude,
                ref_latitude=ref_latitude,
                ref_longitude=ref_longitude)
            if (node_type == 'pole'):
                node_type = "meterhub"
                allocation_capacity = grid.get_default_hub_capacity()

            else:

                node_type = "household"
                allocation_capacity = 0

            grid.add_node(label=str(node_index),
                          x_coordinate=x,
                          y_coordinate=y,
                          node_type=node_type,
                          type_fixed=type_fixed,
                          allocation_capacity=allocation_capacity)
    if grid.get_default_hub_capacity() == 0:
        min_number_of_hubs = 1
    else:
        min_number_of_hubs = (
            int(np.ceil(grid.get_nodes().shape[0]/(1 * grid.get_default_hub_capacity())))
        )

    number_of_hubs = max(opt.get_expected_hub_number_from_k_means(grid=grid),
                         min_number_of_hubs)

    number_of_relaxation_steps_nr = optimize_grid_request.number_of_relaxation_steps_nr

    opt.nr_optimization(grid=grid,
                        number_of_hubs=number_of_hubs,
                        number_of_relaxation_steps=number_of_relaxation_steps_nr,
                        locate_new_hubs_freely=True,
                        first_guess_strategy='random',
                        save_output=False,
                        number_of_hill_climbers_runs=0)

    conn = sqlite3.connect(grid_db)
    sqliteConnection = sqlite3.connect(grid_db)
    cursor = sqliteConnection.cursor()

    # Update nodes types in node database
    for index in grid.get_nodes().index:

        # The indices of the virtual hubs of the nr_optimization method
        # all start with 'V'
        if 'V' in index:
            node_type = 'pole'

            nodes = models.Nodes()

            latitude, longitude = conv.latitude_longitude_from_xy_coordinates(
                x_coord=grid.get_nodes().at[index, "x_coordinate"],
                y_coord=grid.get_nodes().at[index, "y_coordinate"],
                ref_latitude=ref_latitude,
                ref_longitude=ref_longitude)

            nodes.latitude = latitude
            nodes.longitude = longitude
            nodes.area = 0
            nodes.node_type = "pole"
            nodes.fixed_type = False
            nodes.required_capacity = 0
            nodes.max_power = 0
            nodes.is_connected = True

            db.add(nodes)
            db.commit()
        else:
            node_type = grid.get_nodes().at[index, "node_type"]
            if node_type == 'meterhub':
                node_type = 'pole'
            else:
                node_type = "low-demand"
            sql_delete_query = (
                f"""UPDATE nodes
                SET node_type = '{node_type}'
                WHERE  id = {index};
                """)
            cursor.execute(sql_delete_query)
            sqliteConnection.commit()

    # Empty links table
    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()

    # Add newly computed links to database

    cursor = conn.cursor()
    records = []
    count = 1
    for index, row in grid.get_links().iterrows():
        x_from = grid.get_nodes().loc[row['from']]['x_coordinate']
        y_from = grid.get_nodes().loc[row['from']]['y_coordinate']

        x_to = grid.get_nodes().loc[row['to']]['x_coordinate']
        y_to = grid.get_nodes().loc[row['to']]['y_coordinate']

        lat_from, long_from = conv.latitude_longitude_from_xy_coordinates(
            x_coord=x_from,
            y_coord=y_from,
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude
        )

        lat_to, long_to = conv.latitude_longitude_from_xy_coordinates(
            x_coord=x_to,
            y_coord=y_to,
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude
        )

        cable_type = row['type']
        distance = row['distance']

        records.append((count,
                        lat_from,
                        long_from,
                        lat_to,
                        long_to,
                        cable_type,
                        distance))
        count += 1

    cursor.executemany(
        'INSERT INTO links VALUES(?, ?, ?, ?, ?, ?, ?)', records)

    # commit the changes to db
    conn.commit()
    # close the connection
    cursor.close()

    return {
        "code": "success",
        "message": "grid optimized"
    }


@app.post("/shs_identification/")
def identify_shs(shs_identification_request: models.ShsIdentificationRequest,
                 db: Session = Depends(get_db)):

    print("starting shs_identification...")

    res = db.execute("select * from nodes")
    nodes = res.fetchall()

    if len(nodes) == 0:
        return {
            "code": "success",
            "message": "No nodes in table, no identification to be performed"
        }

    # use latitude of the node that is the most west to set origin of x coordinates
    ref_latitude = math.radians(min([node[1] for node in nodes]))
    # use latitude of the node that is the most south to set origin of y coordinates
    ref_longitude = math.radians(min([node[2] for node in nodes]))

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

    sqliteConnection = sqlite3.connect(grid_db)
    conn = sqlite3.connect(grid_db)
    cursor = conn.cursor()

    for index in nodes_df.index:
        if index in nodes_to_disconnect_from_grid:
            sql_delete_query = (
                f"""UPDATE nodes
                SET node_type = 'shs'
                WHERE  id = {index};
                """)
        else:
            sql_delete_query = (
                f"""UPDATE nodes
                SET node_type = 'household'
                WHERE  id = {index};
                """)
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


def clear_single_node(index):
    """
    This function clears the node from the grid.db database.
    """
    sqliteConnection = sqlite3.connect(grid_db)
    cursor = sqliteConnection.cursor()

    sql_delete_query = (
        f"""DELETE FROM nodes WHERE id = {index};"""
    )
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()

    cursor.close()


def clear_nodes_table():
    """
    This function clears the nodes table of the grid.db database.
    """
    sqliteConnection = sqlite3.connect(grid_db)
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from nodes"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()

    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()


@app.post("/clear_node_db/")
async def clear_nodes():
    clear_nodes_table()

    return {
        "code": "success",
        "message": "nodes cleared"
    }


def clear_links_table():
    sqliteConnection = sqlite3.connect(grid_db)
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()


@app.post("/clear_link_db/")
async def clear_links():
    clear_links_table()
    return {
        "code": "success",
        "message": "links cleared"
    }


# -------------------------- FUNCTION FOR DEBUGGING-------------------------- #
def debugging_mode():
    """
    if host="0.0.0.0" and port=8000 does not work, the following can be used:
        host="127.0.0.1", port=8080
    """
    uvicorn.run(app, host="127.0.0.1", port=8080)
