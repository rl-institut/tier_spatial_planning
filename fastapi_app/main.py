from fastapi.param_functions import Query
from fastapi import FastAPI, Request, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from database import SessionLocal, engine
import models
from models import Nodes, AddNodeRequest, OptimizeGridRequest, ValidateBoundariesRequest
from sqlalchemy.orm import Session
import sqlite3
from sgdot.grids import Grid
from sgdot.tools.grid_optimizer import GridOptimizer
import math
import urllib.request
import json
import tools.boundary_identification as bi


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="templates")


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    nodes = db.query(Nodes)
    return templates.TemplateResponse("home.html", {
        "request": request, "nodes": nodes
    })


@app.get("/nodes_db_html")
async def get_nodes(request: Request, db: Session = Depends(get_db)):
    res = db.execute("select * from nodes")
    result = res.fetchall()
    return result


@app.get("/links_db_html")
async def get_links(request: Request, db: Session = Depends(get_db)):
    res = db.execute("select * from links")
    result = res.fetchall()
    return result


@app.post("/validate_boundaries")
async def validate_boundaries(
        validateBoundariesRequest: ValidateBoundariesRequest,
        db: Session = Depends(get_db)):

    boundary_coordinates = validateBoundariesRequest.boundary_coordinates
    latitudes = [x[0] for x in boundary_coordinates]
    longitudes = [x[1] for x in boundary_coordinates]
    min_latitude = min(latitudes)
    min_longitude = min(longitudes)
    max_latitude = max(latitudes)
    max_longitude = max(longitudes)

    url = f'https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500][bbox:{min_latitude},{min_longitude},{max_latitude},{max_longitude}];(way["building"];relation["building"];);out body;>;out skel qt;'
    url_formated = url.replace(" ", "+")
    with urllib.request.urlopen(url_formated) as url:
        data = json.loads(url.read().decode())
    formated_geojson = bi.convert_json_to_polygones_geojson(data)

    building_coord = bi.get_dict_with_mean_coordinate_from_geojson(
        formated_geojson)

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
    for label, coordinates in building_coordidates_within_boundaries.items():
        nodes = Nodes()

        nodes.latitude = coordinates[0]
        nodes.longitude = coordinates[1]
        nodes.node_type = "undefined"
        nodes.fixed_type = False

        db.add(nodes)
        db.commit()

    return formated_geojson


@app.post("/add_node/")
async def add_node(add_node_request: AddNodeRequest,
                   background_tasks: BackgroundTasks,
                   db: Session = Depends(get_db)):
    nodes = Nodes()

    nodes.latitude = add_node_request.latitude
    nodes.longitude = add_node_request.longitude
    nodes.node_type = add_node_request.node_type
    nodes.fixed_type = add_node_request.fixed_type

    db.add(nodes)
    db.commit()

    return {
        "code": "success",
        "message": "node added to db"
    }


@app.post("/optimize_grid/")
async def optimize_grid(optimize_grid_request: OptimizeGridRequest,
                        background_tasks: BackgroundTasks,
                        db: Session = Depends(get_db)):
    # Create GridOptimizer object
    opt = GridOptimizer()

    res = db.execute("select * from nodes")
    nodes = res.fetchall()
    # Create new grid object
    grid = Grid(price_meterhub=optimize_grid_request.price_meterhub,
                price_household=optimize_grid_request.price_household,
                price_interhub_cable_per_meter=optimize_grid_request.price_interhub_cable,
                price_distribution_cable_per_meter=optimize_grid_request.price_distribution_cable)
    # Make sure that new grid object is empty before adding nodes to it
    grid.clear_nodes_and_links()

    r = 6371000     # Radius of the earth [m]
    # use latitude of the node that is the most west to set origin of x coordinates
    latitude_0 = math.radians(min([node[1] for node in nodes]))
    # use latitude of the node that is the most south to set origin of y coordinates
    longitude_0 = math.radians(min([node[2] for node in nodes]))
    for node in nodes:
        latitude = math.radians(node[1])
        longitude = math.radians(node[2])

        x = r * (longitude - longitude_0) * math.cos(latitude_0)
        y = r * (latitude - latitude_0)
        if node[3] == "meterhub":
            node_type = "meterhub"
        else:
            node_type = "household"

        node_type = "household"

        grid.add_node(label=str(node[0]),
                      pixel_x_axis=x,
                      pixel_y_axis=y,
                      node_type=node_type,
                      type_fixed=bool(node[4]))
    number_of_hubs = opt.get_expected_hub_number_from_k_means(grid=grid)
    opt.nr_optimization(grid=grid, number_of_hubs=number_of_hubs, number_of_relaxation_step=10,
                        save_output=False, save_opt_video=False, plot_price_evolution=False)

    sqliteConnection = sqlite3.connect('nodes.db')
    cursor = sqliteConnection.cursor()

    # Empty links table
    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()

    # Update nodes types in node database
    for index in grid.get_nodes().index:
        sql_delete_query = (
            f"""UPDATE nodes
            SET node_type = '{grid.get_nodes().at[index, "node_type"]}'
            WHERE  id = {index};
            """)
        cursor.execute(sql_delete_query)
        sqliteConnection.commit()

    for index in grid.get_hubs().index:
        sql_delete_query = (
            f"""UPDATE nodes
            SET node_type = 'meterhub'
            WHERE  id = {index};
            """)
        cursor.execute(sql_delete_query)
        sqliteConnection.commit()
    cursor.close()

    conn = sqlite3.connect('nodes.db')
    cursor = conn.cursor()
    records = []
    count = 1
    for index, row in grid.get_links().iterrows():
        x_from = grid.get_nodes().loc[row['from']]['pixel_x_axis']
        y_from = grid.get_nodes().loc[row['from']]['pixel_y_axis']

        x_to = grid.get_nodes().loc[row['to']]['pixel_x_axis']
        y_to = grid.get_nodes().loc[row['to']]['pixel_y_axis']

        long_from = math.degrees(longitude_0 + x_from / (r * math.cos(latitude_0)))

        lat_from = math.degrees(latitude_0 + y_from / r)

        long_to = math.degrees(longitude_0 + x_to / (r * math.cos(latitude_0)))

        lat_to = math.degrees(latitude_0 + y_to / r)

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

    cursor.executemany('INSERT INTO links VALUES(?, ?, ?, ?, ?, ?, ?)', records)

    # commit the changes to db
    conn.commit()
    # close the connection
    conn.close()

    return {
        "code": "success",
        "message": "grid optimized"
    }


@ app.post("/clear_node_db/")
async def clear_nodes():
    sqliteConnection = sqlite3.connect('nodes.db')
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from nodes"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()

    sqliteConnection = sqlite3.connect('nodes.db')
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()

    return {
        "code": "success",
        "message": "nodes cleared"
    }


@ app.post("/clear_link_db/")
async def clear_links():
    sqliteConnection = sqlite3.connect('nodes.db')
    cursor = sqliteConnection.cursor()

    sql_delete_query = """DELETE from links"""
    cursor.execute(sql_delete_query)
    sqliteConnection.commit()
    cursor.close()

    return {
        "code": "success",
        "message": "nodes cleared"
    }
