from statistics import mode
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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi_app.database import SessionLocal, engine
from sqlalchemy.orm import Session, raiseload
import sqlite3
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.optimizer import Optimizer, GridOptimizer, EnergySystemOptimizer
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
# import the builtin time module
import time
from datetime import timedelta

# Grab Currrent Time Before Running the Code
start_execution_time = time.monotonic()

app = FastAPI()

app.mount("/fastapi_app/static",
          StaticFiles(directory="fastapi_app/static"), name="static")

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="fastapi_app/pages")

# define different directories for:
# (1) database: *.csv files for nodes and links,
# (2) inputs: input excel files (cost data and timeseries) for offgridders + web app import and export files, and
# (3) outputs: offgridders results
directory_parent = "fastapi_app"

directory_database = os.path.join(directory_parent, 'data', 'database').replace("\\", "/")
full_path_nodes = os.path.join(directory_database, 'nodes.csv').replace("\\", "/")
full_path_links = os.path.join(directory_database, 'links.csv').replace("\\", "/")
full_path_demands = os.path.join(directory_database, 'demands.csv').replace("\\", "/")
full_path_stored_data = os.path.join(directory_database, 'stored_data.csv').replace("\\", "/")
full_path_demand_coverage = os.path.join(
    directory_database, 'demand_coverage.csv').replace("\\", "/")
full_path_energy_flows = os.path.join(
    directory_database, 'energy_flows.csv').replace("\\", "/")
full_path_duration_curves = os.path.join(
    directory_database, 'duration_curves.csv').replace("\\", "/")
full_path_co2_emissions = os.path.join(
    directory_database, 'co2_emissions.csv').replace("\\", "/")
os.makedirs(directory_database, exist_ok=True)

directory_inputs = os.path.join(directory_parent, 'data', 'inputs').replace("\\", "/")
full_path_timeseries = os.path.join(directory_inputs, 'timeseries.csv').replace("\\", "/")
os.makedirs(directory_inputs, exist_ok=True)

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
    response = RedirectResponse(url='/fastapi_app/static/assets/favicon/favicon.ico')
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
    return templates.TemplateResponse("project-setup.html", {
        "request": request
    })


@app.get("/customer_selection")
async def customer_selection(request: Request):
    return templates.TemplateResponse("customer-selection.html", {
        "request": request
    })


@app.get("/energy_system_design")
async def energy_system_design(request: Request):
    return templates.TemplateResponse("energy-system-design.html", {
        "request": request
    })


@app.get("/simulation_results")
async def simulation_results(request: Request):
    return templates.TemplateResponse("simulation-results.html", {
        "request": request
    })


@app.get("/get_demand_coverage_data")
async def get_demand_coverage_data():

    return json.loads(pd.read_csv(full_path_demand_coverage).to_json())


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
        "surface_area",
        "peak_demand",
        "average_consumption",
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
    header_stored_data = [
        "n_consumers",
        "n_poles",
        "length_hv_cable",
        "length_lv_cable",
        "cost_grid",
        "lcoe",
        "res",
        "co2_savings",
        "excess_share",
        "cost_renewable_assets",
        "cost_non_renewable_assets",
        "cost_fuel",
        "pv_capacity",
        "battery_capacity",
        "inverter_capacity",
        "rectifier_capacity",
        "diesel_genset_capacity",
        "peak_demand",
        "surplus",
        "fuel_to_diesel_genset",
        "diesel_genset_to_rectifier",
        "diesel_genset_to_demand",
        "rectifier_to_dc_bus",
        "pv_to_dc_bus",
        "battery_to_dc_bus",
        "dc_bus_to_battery",
        "dc_bus_to_inverter",
        "dc_bus_to_surplus",
        "inverter_to_demand",
        "solver",
        "grid_optimization",
        "time"
    ]
    header_energy_flows = [
        "diesel_genset_production",
        "pv_production",
        "battery_charge",
        "battery_discharge",
        "battery_content",
        "demand",
        "surplus",
    ]
    header_demand_coverage = [
        "demand",
        "renewable",
        "non-renewable",
        "excess"
    ]
    header_duration_curves = [
        "diesel_genset_percentage",
        "diesel_genset_duration",
        "pv_percentage",
        "pv_duration",
        "rectifier_percentage",
        "rectifier_duration",
        "inverter_percentage",
        "inverter_duration",
        "battery_charge_percentage",
        "battery_charge_duration",
        "battery_discharge_percentage",
        "battery_discharge_duration",
    ]
    header_co2_emissions = [
        "non_renewable_electricity_production",
        "hybrid_electricity_production",
        "co2_savings",
    ]

    if nodes:
        pd.DataFrame(columns=header_nodes).to_csv(full_path_nodes, index=False)

    if links:
        pd.DataFrame(columns=header_links).to_csv(full_path_links, index=False)

    pd.DataFrame(columns=header_stored_data).to_csv(full_path_stored_data, index=False)
    pd.DataFrame(columns=header_energy_flows).to_csv(full_path_energy_flows, index=False)
    pd.DataFrame(columns=header_duration_curves).to_csv(full_path_duration_curves, index=False)
    pd.DataFrame(columns=header_co2_emissions).to_csv(full_path_co2_emissions, index=False)
    pd.DataFrame(columns=header_demand_coverage).to_csv(full_path_demand_coverage, index=False)


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
    nodes[headers[5]] = [add_node_request.peak_demand]
    nodes[headers[5]] = [add_node_request.average_consumption]
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
        # newly added nodes
        df = pd.DataFrame.from_dict(nodes).round(decimals=6)
        # the existing database
        df_existing = pd.read_csv(full_path_nodes)
        # append the existing database with the new nodes and remove duplicates (only when both lat and lon are identical)
        df_total = df_existing.append(df).drop_duplicates(
            subset=['latitude', 'longitude'], inplace=False)

        # nodes obtained from a previous optimization (e.g., poles)
        # will not be considered in the grid optimization
        for node_index in df_total.index:
            if ('nr-optimization' in df_total.how_added[node_index]):
                df_total.drop(labels=node_index, axis=0, inplace=True)
        # storing the nodes in the database (updating the existing CSV file)
        df_total = df_total.reset_index(drop=True)

        # defining the precision of data
        df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
        df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
        df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
        df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
        df_total.average_consumption = df_total.average_consumption.map(lambda x: "%.3f" % x)

        # finally adding the refined dataframe (if it is not empty) to the existing csv file
        if len(df_total.index) != 0:
            df_total.to_csv(full_path_nodes, index=False,
                            header=df_existing.head)
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


# # remove some nodes from the database
# def database_remove_nodes(nodes):

#     nodes_index_removing = []
#     for node_index in nodes.index:
#         if ('nr-optimization' in nodes.how_added[node_index]):
#             nodes_index_removing.append(node_index)

#     for index in nodes.index:
#         if index in nodes_index_removing:
#             nodes.drop(labels=index, axis=0, inplace=True)

#     # storing the nodes in the database (updating the existing CSV file)
#     nodes = nodes.reset_index(drop=True)
#     database_add(add_nodes=True, add_links=False, inlet=nodes.to_dict())


@app.get("/database_to_js/{nodes_or_links}")
async def database_read(nodes_or_links: str):

    # importing nodes and links from the csv files to the map
    if nodes_or_links == 'nodes':
        nodes_list = json.loads(pd.read_csv(full_path_nodes).to_json())
        return nodes_list
    else:
        links_list = json.loads(pd.read_csv(full_path_links).to_json())
        return links_list


@app.get("/load_results")
async def load_results():

    results = {}

    df = pd.read_csv(full_path_stored_data)

    results['n_poles'] = str(df.loc[0, 'n_poles'])
    results['n_consumers'] = str(df.loc[0, 'n_consumers'])
    results['length_hv_cable'] = str(df.loc[0, 'length_hv_cable']) + ' m'
    results['length_lv_cable'] = str(df.loc[0, 'length_lv_cable']) + ' m'
    results['cost_grid'] = str(df.loc[0, 'cost_grid']) + ' EUR/a'
    results['lcoe'] = str(df.loc[0, 'lcoe']) + ' c/kWh'
    results['res'] = str(df.loc[0, 'res']) + ' %'
    results['co2_savings'] = str(df.loc[0, 'co2_savings']) + ' t/a'
    results['excess_share'] = str(df.loc[0, 'excess_share']) + ' %'
    results['solver'] = str(df.loc[0, 'solver']).title()
    results['grid_optimization'] = str(df.loc[0, 'grid_optimization'])
    results['time'] = str(df.loc[0, 'time']) + ' s'

    # importing nodes and links from the csv files to the map
    return results


@app.get("/get_optimal_capacities")
async def get_optimal_capacities():

    optimal_capacities = {}

    df = pd.read_csv(full_path_stored_data)

    optimal_capacities['pv'] = str(df.loc[0, 'pv_capacity'])
    optimal_capacities['battery'] = str(df.loc[0, 'battery_capacity'])
    optimal_capacities['inverter'] = str(df.loc[0, 'inverter_capacity'])
    optimal_capacities['rectifier'] = str(df.loc[0, 'rectifier_capacity'])
    optimal_capacities['diesel_genset'] = str(df.loc[0, 'diesel_genset_capacity'])
    optimal_capacities['peak_demand'] = str(df.loc[0, 'peak_demand'])
    optimal_capacities['surplus'] = str(df.loc[0, 'surplus'])

    # importing nodes and links from the csv files to the map
    return optimal_capacities


@app.get("/get_lcoe_breakdown")
async def get_lcoe_breakdown():

    lcoe_breakdown = {}

    df = pd.read_csv(full_path_stored_data)

    lcoe_breakdown['renewable_assets'] = str(df.loc[0, 'cost_renewable_assets'])
    lcoe_breakdown['non_renewable_assets'] = str(df.loc[0, 'cost_non_renewable_assets'])
    lcoe_breakdown['grid'] = str(df.loc[0, 'cost_grid'])
    lcoe_breakdown['fuel'] = str(df.loc[0, 'cost_fuel'])

    # importing nodes and links from the csv files to the map
    return lcoe_breakdown


@app.get("/get_data_for_sankey_diagram")
async def get_data_for_sankey_diagram():

    sankey_data = {}

    df = pd.read_csv(full_path_stored_data)

    sankey_data['fuel_to_diesel_genset'] = str(df.loc[0, 'fuel_to_diesel_genset'])
    sankey_data['diesel_genset_to_rectifier'] = str(df.loc[0, 'diesel_genset_to_rectifier'])
    sankey_data['diesel_genset_to_demand'] = str(df.loc[0, 'diesel_genset_to_demand'])
    sankey_data['rectifier_to_dc_bus'] = str(df.loc[0, 'rectifier_to_dc_bus'])
    sankey_data['pv_to_dc_bus'] = str(df.loc[0, 'pv_to_dc_bus'])
    sankey_data['battery_to_dc_bus'] = str(df.loc[0, 'battery_to_dc_bus'])
    sankey_data['dc_bus_to_battery'] = str(df.loc[0, 'dc_bus_to_battery'])
    sankey_data['dc_bus_to_inverter'] = str(df.loc[0, 'dc_bus_to_inverter'])
    sankey_data['dc_bus_to_surplus'] = str(df.loc[0, 'dc_bus_to_surplus'])
    sankey_data['inverter_to_demand'] = str(df.loc[0, 'inverter_to_demand'])

    # importing nodes and links from the csv files to the map
    return sankey_data


@app.get("/get_data_for_energy_flows")
async def get_data_for_energy_flows():

    return json.loads(pd.read_csv(full_path_energy_flows).to_json())


@app.get("/get_data_for_duration_curves")
async def get_data_for_duration_curves():

    return json.loads(pd.read_csv(full_path_duration_curves).to_json())


@app.get("/get_co2_emissions_data")
async def get_co2_emissions_data():

    return json.loads(pd.read_csv(full_path_co2_emissions).to_json())


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
        # to the 'database_add' function to store nodes properties in the database
        nodes = defaultdict(list)
        for label, coordinates in building_coordidates_within_boundaries.items():
            nodes["latitude"].append(coordinates[0])
            nodes["longitude"].append(coordinates[1])
            nodes["node_type"].append("consumer")
            nodes["consumer_type"].append("household")
            nodes["consumer_detail"].append("default")

            # surface area is taken from the open street map
            nodes["surface_area"].append(building_area[label])

        # after collecting all surface areas, based on a simple assumption, the peak demand will be obtained
        max_surface_area = max(nodes['surface_area'])

        # calculate the total peak demand for each of the five demand profiles to make the final demand profile
        peak_very_low_demand = 0
        peak_low_demand = 0
        peak_medium_demand = 0
        peak_high_demand = 0
        peak_very_high_demand = 0

        for area in nodes['surface_area']:
            if area <= 0.2 * max_surface_area:
                nodes['peak_demand'].append(0.01 * area)
                peak_very_low_demand += (0.01 * area)
            elif area < 0.4 * max_surface_area:
                nodes['peak_demand'].append(0.02 * area)
                peak_low_demand += (0.02 * area)
            elif area < 0.6 * max_surface_area:
                nodes['peak_demand'].append(0.03 * area)
                peak_medium_demand += (0.03 * area)
            elif area < 0.8 * max_surface_area:
                nodes['peak_demand'].append(0.04 * area)
                peak_high_demand += (0.04 * area)
            else:
                nodes['peak_demand'].append(0.05 * area)
                peak_very_high_demand += (0.05 * area)

        # normalized demands is a CSV file with 5 columns representing the very low to very high demand profiles
        normalized_demands = pd.read_csv(full_path_demands, delimiter=';', header=None)

        max_peak_demand = max(nodes['peak_demand'])
        counter = 0
        for peak_demand in nodes['peak_demand']:
            if peak_demand <= 0.2 * max_peak_demand:
                nodes['average_consumption'].append(
                    normalized_demands.iloc[:, 0].sum() * nodes['peak_demand'][counter])
            elif peak_demand < 0.4 * max_peak_demand:
                nodes['average_consumption'].append(
                    normalized_demands.iloc[:, 1].sum() * nodes['peak_demand'][counter])
            elif peak_demand < 0.6 * max_peak_demand:
                nodes['average_consumption'].append(
                    normalized_demands.iloc[:, 2].sum() * nodes['peak_demand'][counter])
            elif peak_demand < 0.8 * max_peak_demand:
                nodes['average_consumption'].append(
                    normalized_demands.iloc[:, 3].sum() * nodes['peak_demand'][counter])
            else:
                nodes['average_consumption'].append(
                    normalized_demands.iloc[:, 4].sum() * nodes['peak_demand'][counter])

            counter += 1

            # it is assumed that all nodes are parts of the mini-grid
            # later, when the shs candidates are obtained, the corresponding
            # values will be changed to 'False'
            nodes["is_connected"].append(True)

            # the node is selected automatically after drawing boundaries
            nodes["how_added"].append("automatic")

        # storing the nodes in the database
        database_add(add_nodes=True, add_links=False, inlet=nodes)

        # create the total demand profile of the selected buildings
        total_demand = normalized_demands.iloc[:, 0] * peak_very_low_demand + normalized_demands.iloc[:, 1] * peak_low_demand + \
            normalized_demands.iloc[:, 2] * peak_medium_demand + normalized_demands.iloc[:, 3] * peak_high_demand + \
            normalized_demands.iloc[:, 4] * peak_very_high_demand

        # load timeseries data
        timeseries = pd.read_csv(full_path_timeseries)

        # replace the demand column in the timeseries file with the total demand calculated here
        timeseries['Demand'] = total_demand

        # update the CSV file
        timeseries.to_csv(full_path_timeseries, index=False)

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
    opt = GridOptimizer(start_date=optimize_grid_request.start_date,
                        n_days=optimize_grid_request.n_days,
                        project_lifetime=optimize_grid_request.project_lifetime,
                        wacc=optimize_grid_request.wacc,
                        tax=optimize_grid_request.tax)

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
    await database_initialization(nodes=False, links=True)

    # create a new "grid" object from the Grid class
    epc_hv_cable = (opt.crf * Optimizer.capex_multi_investment(
        opt,
        capex_0=optimize_grid_request.hv_cable['capex'],
        component_lifetime=optimize_grid_request.hv_cable['lifetime']
    ) + optimize_grid_request.hv_cable['opex']) * opt.n_days / 365

    epc_lv_cable = (opt.crf * Optimizer.capex_multi_investment(
        opt,
        capex_0=optimize_grid_request.lv_cable['capex'],
        component_lifetime=optimize_grid_request.lv_cable['lifetime']
    ) + optimize_grid_request.lv_cable['opex']) * opt.n_days / 365

    epc_connection = (opt.crf * Optimizer.capex_multi_investment(
        opt,
        capex_0=optimize_grid_request.connection['capex'],
        component_lifetime=optimize_grid_request.connection['lifetime']
    ) + optimize_grid_request.connection['opex']) * opt.n_days / 365

    epc_pole = (opt.crf * Optimizer.capex_multi_investment(
        opt,
        capex_0=optimize_grid_request.pole['capex'],
        component_lifetime=optimize_grid_request.pole['lifetime']
    ) + optimize_grid_request.pole['opex']) * opt.n_days / 365

    grid = Grid(
        epc_hv_cable=epc_hv_cable,
        epc_lv_cable=epc_lv_cable,
        epc_connection=epc_connection,
        epc_pole=epc_pole,
        pole_max_connection=optimize_grid_request.pole['max_connections']
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

    number_of_relaxation_steps_nr = optimize_grid_request.optimization['n_relaxation_steps']

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

    # store data for showing in the final results
    df = pd.read_csv(full_path_stored_data)
    df.loc[0, 'n_consumers'] = len(grid.consumers())
    df.loc[0, 'n_poles'] = len(grid.poles())
    df.loc[0, 'length_hv_cable'] = int(
        grid.links[grid.links.link_type == 'interpole']['length'].sum())
    df.loc[0, 'length_lv_cable'] = int(
        grid.links[grid.links.link_type == 'distribution']['length'].sum())
    df.loc[0, 'cost_grid'] = int(grid.cost())
    df.loc[0, 'grid_optimization'] = 'NR'
    df.to_csv(full_path_stored_data, mode='a', header=False, index=False, float_format='%.0f')


@ app.post('/optimize_energy_system')
async def optimize_energy_system(optimize_energy_system_request: models.OptimizeEnergySystemRequest):
    ensys_opt = EnergySystemOptimizer(
        start_date=optimize_energy_system_request.start_date,
        n_days=optimize_energy_system_request.n_days,
        project_lifetime=optimize_energy_system_request.project_lifetime,
        wacc=optimize_energy_system_request.wacc,
        tax=optimize_energy_system_request.tax,
        path_data=full_path_timeseries,
        solver='gurobi',
        pv=optimize_energy_system_request.pv,
        diesel_genset=optimize_energy_system_request.diesel_genset,
        battery=optimize_energy_system_request.battery,
        inverter=optimize_energy_system_request.inverter,
        rectifier=optimize_energy_system_request.rectifier,
    )
    ensys_opt.optimize_energy_system()

    # unit for co2_emission_factor is kgCO2 per kWh of produced electricity
    if ensys_opt.capacity_genset < 60:
        co2_emission_factor = 1.580
    elif ensys_opt.capacity_genset < 300:
        co2_emission_factor = 0.883
    else:
        co2_emission_factor = 0.699

    # store fuel co2 emissions (kg_CO2 per L of fuel)
    df = pd.read_csv(full_path_co2_emissions)
    df.loc[:, 'non_renewable_electricity_production'] = np.cumsum(
        ensys_opt.demand) * co2_emission_factor / 1000  # tCO2 per year
    df.loc[:, 'hybrid_electricity_production'] = np.cumsum(
        ensys_opt.sequences_genset) * co2_emission_factor / 1000  # tCO2 per year
    df.loc[:, 'co2_savings'] = df.loc[:, 'non_renewable_electricity_production'] - \
        df.loc[:, 'hybrid_electricity_production']  # tCO2 per year
    df.to_csv(full_path_co2_emissions, index=False, float_format='%.3f')
    # TODO: -2 must actually be -1, but for some reason, the co2-emission csv file has an additional empty row
    co2_savings = df.loc[:, 'co2_savings'][-2]  # takes the last element of the cumulative sum

    # store data for showing in the final results
    df = pd.read_csv(full_path_stored_data)
    df.loc[0, 'cost_renewable_assets'] = ensys_opt.total_renewable
    df.loc[0, 'cost_non_renewable_assets'] = ensys_opt.total_non_renewable
    df.loc[0, 'cost_fuel'] = ensys_opt.total_fuel
    df.loc[0, 'lcoe'] = 100 * (ensys_opt.total_revenue +
                               df.loc[0, 'cost_grid']) / ensys_opt.total_demand
    df.loc[0, 'res'] = ensys_opt.res
    df.loc[0, 'co2_savings'] = co2_savings
    df.loc[0, 'excess_share'] = ensys_opt.excess_rate
    df.loc[0, 'pv_capacity'] = ensys_opt.capacity_pv
    df.loc[0, 'battery_capacity'] = ensys_opt.capacity_battery
    df.loc[0, 'inverter_capacity'] = ensys_opt.capacity_inverter
    df.loc[0, 'rectifier_capacity'] = ensys_opt.capacity_rectifier
    df.loc[0, 'diesel_genset_capacity'] = ensys_opt.capacity_genset
    df.loc[0, 'peak_demand'] = ensys_opt.demand_peak
    df.loc[0, 'surplus'] = ensys_opt.sequences_excess.max()
    # data for sankey diagram - all in MWh
    df.loc[0, 'fuel_to_diesel_genset'] = ensys_opt.sequences_fuel_consumption.sum() * 0.846 * \
        ensys_opt.diesel_genset['fuel_lhv'] / 1000
    df.loc[0, 'diesel_genset_to_rectifier'] = ensys_opt.sequences_rectifier.sum() / \
        ensys_opt.rectifier['efficiency'] / 1000
    df.loc[0, 'diesel_genset_to_demand'] = ensys_opt.sequences_genset.sum() / 1000 - \
        df.loc[0, 'diesel_genset_to_rectifier']
    df.loc[0, 'rectifier_to_dc_bus'] = ensys_opt.sequences_rectifier.sum() / 1000
    df.loc[0, 'pv_to_dc_bus'] = ensys_opt.sequences_pv.sum() / 1000
    df.loc[0, 'battery_to_dc_bus'] = ensys_opt.sequences_battery_discharge.sum() / 1000
    df.loc[0, 'dc_bus_to_battery'] = ensys_opt.sequences_battery_charge.sum() / 1000
    df.loc[0, 'dc_bus_to_inverter'] = ensys_opt.sequences_inverter.sum() / \
        ensys_opt.inverter['efficiency'] / 1000
    df.loc[0, 'dc_bus_to_surplus'] = ensys_opt.sequences_excess.sum() / 1000
    df.loc[0, 'inverter_to_demand'] = ensys_opt.sequences_inverter.sum() / 1000
    df.loc[0, 'solver'] = ensys_opt.solver
    # Grab Currrent Time After Running the Code
    end_execution_time = time.monotonic()
    df.loc[0, 'time'] = end_execution_time - start_execution_time
    df.to_csv(full_path_stored_data, index=False, float_format='%.1f')

    # store energy flows
    df = pd.read_csv(full_path_energy_flows)
    df.loc[:, 'diesel_genset_production'] = ensys_opt.sequences_genset
    df.loc[:, 'pv_production'] = ensys_opt.sequences_pv
    df.loc[:, 'battery_charge'] = ensys_opt.sequences_battery_charge
    df.loc[:, 'battery_discharge'] = ensys_opt.sequences_battery_discharge
    df.loc[:, 'battery_content'] = ensys_opt.sequences_battery_content
    df.loc[:, 'demand'] = ensys_opt.sequences_demand
    df.loc[:, 'surplus'] = ensys_opt.sequences_excess
    df.to_csv(full_path_energy_flows, index=True, index_label='time', float_format='%.3f')

    # store demand coverage
    df = pd.read_csv(full_path_demand_coverage)
    df.loc[:, 'demand'] = ensys_opt.sequences_demand
    df.loc[:, 'renewable'] = ensys_opt.sequences_inverter
    df.loc[:, 'non_renewable'] = ensys_opt.sequences_genset
    df.loc[:, 'excess'] = ensys_opt.sequences_excess
    df.to_csv(full_path_demand_coverage, index=False, float_format='%.3f')

    # store duration curves
    df = pd.read_csv(full_path_duration_curves)
    df.loc[:, 'diesel_genset_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_genset) + 1) / len(ensys_opt.sequences_genset)
    df.loc[:, 'diesel_genset_duration'] = 100 * np.sort(ensys_opt.sequences_genset)[
        ::-1] / ensys_opt.sequences_genset.max()
    df.loc[:, 'pv_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_pv) + 1) / len(ensys_opt.sequences_pv)
    df.loc[:, 'pv_duration'] = 100 * \
        np.sort(ensys_opt.sequences_pv)[::-1] / ensys_opt.sequences_pv.max()
    df.loc[:, 'rectifier_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_rectifier) + 1) / len(ensys_opt.sequences_rectifier)
    df.loc[:, 'rectifier_duration'] = 100 * np.sort(ensys_opt.sequences_rectifier)[
        ::-1] / ensys_opt.sequences_rectifier.max()
    df.loc[:, 'inverter_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_inverter) + 1) / len(ensys_opt.sequences_inverter)
    df.loc[:, 'inverter_duration'] = 100 * np.sort(ensys_opt.sequences_inverter)[
        ::-1] / ensys_opt.sequences_inverter.max()
    df.loc[:, 'battery_charge_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_battery_charge) + 1) / \
        len(ensys_opt.sequences_battery_charge)
    df.loc[:, 'battery_charge_duration'] = 100 * np.sort(ensys_opt.sequences_battery_charge)[
        ::-1] / ensys_opt.sequences_battery_charge.max()
    df.loc[:, 'battery_discharge_percentage'] = 100 * \
        np.arange(1, len(ensys_opt.sequences_battery_discharge) + 1) / \
        len(ensys_opt.sequences_battery_discharge)
    df.loc[:, 'battery_discharge_duration'] = 100 * np.sort(ensys_opt.sequences_battery_discharge)[
        ::-1] / ensys_opt.sequences_battery_discharge.max()
    df.to_csv(full_path_duration_curves, index=False, float_format='%.3f')


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
