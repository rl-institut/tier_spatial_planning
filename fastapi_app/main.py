import fastapi_app.tools.boundary_identification as bi
import fastapi_app.tools.coordinates_conversion as conv
import fastapi_app.tools.shs_identification as shs_ident
import fastapi_app.db.models as models
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.optimizer import Optimizer, GridOptimizer, EnergySystemOptimizer, po
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi_app.tools.accounts import Hasher, create_guid, is_valid_credentials, send_activation_link, activate_mail, \
    authenticate_user, create_access_token
from fastapi_app.tools import accounts
from fastapi_app.db import config
from fastapi_app.db.database import get_db, get_async_db
from fastapi_app.db import queries, inserts
import math
import urllib.request
import ssl
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# for appending to the dictionary
from collections import defaultdict

# for sending an array of data from JS to the fastAPI
from typing import Any, Dict, List, Union

# import the builtin time module
import time

app = FastAPI()

app.mount("/fastapi_app/static", StaticFiles(directory="fastapi_app/static"), name="static")

templates = Jinja2Templates(directory="fastapi_app/pages")

# define different directories for:
# (1) database: *.csv files for nodes and links,
# (2) inputs: input excel files (cost data and timeseries) for offgridders + web app import and export files, and
# (3) outputs: offgridders results
directory_parent = "fastapi_app"

directory_database = os.path.join(directory_parent, "data", "database").replace("\\", "/")
full_path_demands = os.path.join(directory_database, "demands.csv").replace("\\", "/")
os.makedirs(directory_database, exist_ok=True)
directory_inputs = os.path.join(directory_parent, "data", "inputs").replace("\\", "/")
full_path_timeseries = os.path.join(directory_inputs, "timeseries.csv").replace("\\", "/")
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
    """Redirects request to location of favicon.ico logo in static folder"""
    response = RedirectResponse(url="/fastapi_app/static/assets/favicon/favicon.ico")
    return response


# ************************************************************/
# *                     IMPORT / EXPORT                      */
# ************************************************************/


@app.post("/export_data/")
async def export_data(generate_export_file_request: models.GenerateExportFileRequest):
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
    nodes = await database_read(nodes_or_links="nodes")
    links = await database_read(nodes_or_links="links")
    nodes_df = pd.DataFrame(nodes)
    links_df = pd.DataFrame(links)

    # get all settings defined in the web app
    settings = [element for element in generate_export_file_request]
    settings_df = pd.DataFrame(
        {"Setting": [x[0] for x in settings], "value": [x[1] for x in settings]}
    ).set_index("Setting")

    # create the *.xlsx file with sheets for nodes, links and settings
    with pd.ExcelWriter(
            full_path_import_export
    ) as writer:  # pylint: disable=abstract-class-instantiated
        nodes_df.to_excel(
            excel_writer=writer,
            sheet_name="nodes",
            header=nodes_df.columns,
            index=False,
        )
        links_df.to_excel(
            excel_writer=writer,
            sheet_name="links",
            header=links_df.columns,
            index=False,
        )
        settings_df.to_excel(excel_writer=writer, sheet_name="settings")

    # TO DO: formatting of the excel file


@app.get("/download_export_file",
         responses={200: {"description": "xlsx file containing the information about the configuration.",
                          "content": {"static/io/test_excel_node.xlsx": {"example": "No example available."}}, }}, )
async def download_export_file():
    file_name = "temp.xlsx"
    # Download xlsx file
    file_path = os.path.join(directory_parent, f"import_export/{file_name}")
    if os.path.exists(file_path):
        return FileResponse(path=file_path,
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            filename="backup.xlsx", )
    else:
        return {"error": "File not found!"}


@app.post("/import_data/{project_id}")
async def import_data(project_id, request: Request, import_files: import_structure = None,
                      db: Session = Depends(get_async_db)):
    # add nodes from the 'nodes' sheet of the excel file to the 'nodes.csv' file
    # TODO: update the template for adding nodes
    nodes = import_files["nodes_to_import"]
    links = import_files["links_to_import"]
    user_id = await accounts.get_user_from_cookie(request, db).id
    if len(nodes) > 0:
        await inserts.update_nodes_and_links(True, False, nodes, user_id, project_id, db)

    if len(links) > 0:
        await inserts.update_nodes_and_links(False, True, links, user_id, project_id, db)

    # ------------------------------ HANDLE REQUEST ------------------------------#


@app.get("/")
async def home(request: Request, db: Session = Depends(get_async_db)):
    # return templates.TemplateResponse("project-setup.html", {"request": request})
    user = await accounts.get_user_from_cookie(request, db)
    if user is None:
        return templates.TemplateResponse("landing-page.html", {"request": request})
    else:
        projects = await queries.get_project_of_user(user.id, db)
        for project in projects:
            project.created_at = project.created_at.date()
            project.updated_at = project.updated_at.date()
        return templates.TemplateResponse("user_projects.html", {"request": request, 'projects': projects})


@app.get("/project_setup")
async def project_setup(request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db) # .id
    project_id = request.query_params.get('project_id')
    if project_id is None:
        project_id = queries.next_project_id_of_user(user.id, db)
    return templates.TemplateResponse("project-setup.html", {"request": request, 'project_id': project_id})


@app.get("/user_registration")
async def user_registration(request: Request):
    return templates.TemplateResponse("user-registration.html", {"request": request})


@app.get("/activation_mail/{guid}")
async def activation_mail(request: Request, db: Session = Depends(get_db)):
    guid = request.path_params.get('guid')
    if guid is not None:
        activate_mail(guid[5:], db)
    return templates.TemplateResponse("landing-page.html", {"request": request})


@app.get("/account_overview")
async def account_overview(request: Request, db: Session = Depends(get_db)):
    user = await accounts.get_user_from_cookie(request, db)
    if user is None:
        return templates.TemplateResponse("landing-page.html", {"request": request})
    else:
        return templates.TemplateResponse("account_overview.html", {"request": request})


@app.get("/consumer_selection")
async def consumer_selection(request: Request):
    project_id = request.query_params.get('project_id')
    try:
        int(project_id)
    except (TypeError, ValueError):
        return templates.TemplateResponse("landing-page.html", {"request": request})
    return templates.TemplateResponse("consumer-selection.html", {"request": request, 'project_id': project_id})


@app.get("/grid_design")
async def grid_design(request: Request):
    project_id = request.query_params.get('project_id')
    return templates.TemplateResponse("grid-design.html", {"request": request, 'project_id': project_id})


@app.get("/demand_estimation")
async def demand_estimation(request: Request):
    return templates.TemplateResponse("demand_estimation.html", {"request": request})


@app.get("/energy_system_design")
async def energy_system_design(request: Request):
    project_id = request.query_params.get('project_id')
    return templates.TemplateResponse("energy-system-design.html", {"request": request, 'project_id': project_id})


@app.get("/simulation_results")
async def simulation_results(request: Request):
    project_id = request.query_params.get('project_id')
    try:
        int(project_id)
    except (TypeError, ValueError):
        return templates.TemplateResponse("landing-page.html", {"request": request})
    return templates.TemplateResponse("simulation-results.html", {"request": request, 'project_id': project_id})


@app.get("/calculating")
async def calculating(request: Request, db: Session = Depends(get_async_db)):
    project_id = request.query_params.get('project_id')
    user = await accounts.get_user_from_cookie(request, db)
    try:
        int(project_id)
    except (TypeError, ValueError):
        return templates.TemplateResponse("landing-page.html", {"request": request})
    if 'anonymous' in user.email:
        msg = 'You will be forwarded after the model calculation is completed.'
    else:
        msg = 'You will be forwarded after the model calculation is completed. You can also close the window and view' \
              ' the results in your user account after the calculation is finished. You will be notified by email' \
              ' about the completion of the calculation.'
    return templates.TemplateResponse("calculating.html", {"request": request, 'project_id': project_id, 'msg':msg})


@app.get("/get_demand_coverage_data/{project_id}")
async def get_demand_coverage_data(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_demand_coverage_df(user.id, project_id, db)
    df = df.reset_index(drop=True)
    return json.loads(df.to_json())


@app.get("/database_to_js/{nodes_or_links}/{project_id}")
async def database_read(nodes_or_links: str, project_id, request: Request, db: Session = Depends(get_async_db)):
    # importing nodes and links from the csv files to the map
    user = await accounts.get_user_from_cookie(request, db)
    if nodes_or_links == "nodes":
        nodes_json = await queries.get_nodes_json(user.id, project_id)
        return nodes_json
    else:
        links_json = await queries.get_links_json(user.id, project_id)
        return links_json


@app.get("/load_results/{project_id}")
async def load_results(project_id, request: Request, db: Session = Depends(get_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_results_df(user.id, project_id, db)
    df["average_length_distribution_cable"] = df["length_distribution_cable"] / df["n_distribution_links"]
    df["average_length_connection_cable"] = df["length_connection_cable"] / df["n_connection_links"]
    df["time"] = df["time_grid_design"] + df["time_energy_system_design"]
    unit_dict = {'n_poles': '',
                 'n_consumers': '',
                 'n_shs_consumers': '',
                 'length_distribution_cable': 'm',
                 'average_length_distribution_cable': 'm',
                 'length_connection_cable': 'm',
                 'average_length_connection_cable': 'm',
                 'cost_grid': 'USD/a',
                 'lcoe': 'c/kWh',
                 'res': '%',
                 'shortage_total': '%',
                 'surplus_rate': '%',
                 'time': 's',
                 'co2_savings': 't/a'}
    df = df[list(unit_dict.keys())].round(1).astype(str)
    for col in df.columns:
        df[col] = df[col] + ' ' + unit_dict[col]
    results = df.to_dict(orient='records')[0]
    # importing nodes and links from the csv files to the map
    return results


@app.get("/load_previous_data/{page_name}")
async def load_previous_data(page_name, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    project_id = request.query_params.get('project_id')
    if page_name == "project_setup":
        if project_id == 'new':
            project_id = queries.next_project_id_of_user(user.id, db)
            return models.ProjectSetup(project_id=project_id)
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        project_setup = await queries.get_project_setup_of_user(user.id, project_id, db)
        if hasattr(project_setup, 'start_date'):
            project_setup.start_date = str(project_setup.start_date.date())
            return project_setup
        else:
            return None
    elif page_name == "grid_design":
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        grid_design = await queries.get_grid_design_of_user(user.id, project_id, db)
        return grid_design


@app.post("/add_user_to_db/")
async def add_user_to_db(user: models.Credentials, db: Session = Depends(get_db)):
    res = is_valid_credentials(user, db)
    if res[0] is True:
        guid = create_guid()
        user = models.User(email=user.email,
                           hashed_password=Hasher.get_password_hash(user.password),
                           guid=guid,
                           is_confirmed=False,
                           is_active=False,
                           is_superuser=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        send_activation_link(user.email, guid)
    return models.ValidRegistration(validation=res[0], msg=res[1])


@app.post("/set_access_token/", response_model=models.Token)
async def set_access_token(response: Response, credentials: models.Credentials, db: Session = Depends(get_async_db)):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        user = await authenticate_user(credentials.email, credentials.password, db)
        name = user.email
    else:
        name = 'anonymous'
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": name}, expires_delta=access_token_expires)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}",
                        httponly=True)  # set HttpOnly cookie in response
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/login/")
async def login(response: Response, credentials: models.Credentials, db: Session = Depends(get_async_db)):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        user = await authenticate_user(credentials.email, credentials.password, db)
        del credentials
        if user is not False:
            access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
            response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
        return {"access_token": access_token, "token_type": "bearer"}


@app.post("/logout/")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "success"}


@app.post("/query_account_data/")
async def query_account_data(request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    if user is not None:
        return models.UserOverview(email=user.email)
    else:
        return models.UserOverview(email="")


@app.post("/has_cookie/")
async def has_cookie(request: Request):
    token = request.cookies.get("access_token")
    if token is not None:
        return True
    else:
        return False


@app.post("/save_previous_data/{page_name}")
async def save_previous_data(request: Request, page_name: str, save_previous_data_request:
models.SavePreviousDataRequest, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    project_id = get_project_id_from_request(request)
    if "project_setup" in page_name:
        save_previous_data_request.page_setup['created_at'] = pd.Timestamp.now()
        save_previous_data_request.page_setup['updated_at'] = pd.Timestamp.now()
        save_previous_data_request.page_setup['id'] = user.id
        # ToDo: Raise Error if project id missing redirect
        save_previous_data_request.page_setup['project_id'] = project_id
        project_setup = models.ProjectSetup(**save_previous_data_request.page_setup)
        await db.merge(project_setup)
        await db.commit()
    elif "grid_design" in page_name:
        save_previous_data_request.grid_design['id'] = user.id
        # ToDo: Raise Error if project id missing redirect
        save_previous_data_request.grid_design['project_id'] = project_id
        grid_design = models.GridDesign(**save_previous_data_request.grid_design)
        await db.merge(grid_design)
        await db.commit()


def get_project_id_from_request(request: Request):
    project_id = request.query_params.get('project_id')
    if project_id is None:
        project_id = [tup[1].decode() for tup in request.scope['headers']
                      if 'project_id' in tup[1].decode()][0].split('=')[-1]
    project_id = int(project_id)
    return project_id


@app.get("/get_optimal_capacities/{project_id}")
async def get_optimal_capacities(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_results_df(user.id, project_id, db)
    optimal_capacities = {}
    optimal_capacities["pv"] = str(df.loc[0, "pv_capacity"])
    optimal_capacities["battery"] = str(df.loc[0, "battery_capacity"])
    optimal_capacities["inverter"] = str(df.loc[0, "inverter_capacity"])
    optimal_capacities["rectifier"] = str(df.loc[0, "rectifier_capacity"])
    optimal_capacities["diesel_genset"] = str(df.loc[0, "diesel_genset_capacity"])
    optimal_capacities["peak_demand"] = str(df.loc[0, "peak_demand"])
    optimal_capacities["surplus"] = str(df.loc[0, "surplus"])
    # importing nodes and links from the csv files to the map
    return optimal_capacities


@app.get("/get_lcoe_breakdown/{project_id}")
async def get_lcoe_breakdown(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_results_df(user.id, project_id, db)
    lcoe_breakdown = {}
    lcoe_breakdown["renewable_assets"] = str(df.loc[0, "cost_renewable_assets"])
    lcoe_breakdown["non_renewable_assets"] = str(df.loc[0, "cost_non_renewable_assets"])
    lcoe_breakdown["grid"] = str(df.loc[0, "cost_grid"])
    lcoe_breakdown["fuel"] = str(df.loc[0, "cost_fuel"])
    # importing nodes and links from the csv files to the map
    return lcoe_breakdown


@app.get("/get_data_for_sankey_diagram/{project_id}")
async def get_data_for_sankey_diagram(project_id, request: Request, db: Session = Depends(get_async_db)):
    sankey_data = {}
    user = await  accounts.get_user_from_cookie(request, db)
    df = await queries.get_results_df(user.id, project_id, db)
    sankey_data["fuel_to_diesel_genset"] = str(df.loc[0, "fuel_to_diesel_genset"])
    sankey_data["diesel_genset_to_rectifier"] = str(df.loc[0, "diesel_genset_to_rectifier"])
    sankey_data["diesel_genset_to_demand"] = str(df.loc[0, "diesel_genset_to_demand"])
    sankey_data["rectifier_to_dc_bus"] = str(df.loc[0, "rectifier_to_dc_bus"])
    sankey_data["pv_to_dc_bus"] = str(df.loc[0, "pv_to_dc_bus"])
    sankey_data["battery_to_dc_bus"] = str(df.loc[0, "battery_to_dc_bus"])
    sankey_data["dc_bus_to_battery"] = str(df.loc[0, "dc_bus_to_battery"])
    sankey_data["dc_bus_to_inverter"] = str(df.loc[0, "dc_bus_to_inverter"])
    sankey_data["dc_bus_to_surplus"] = str(df.loc[0, "dc_bus_to_surplus"])
    sankey_data["inverter_to_demand"] = str(df.loc[0, "inverter_to_demand"])
    # importing nodes and links from the csv files to the map
    return sankey_data


@app.get("/get_data_for_energy_flows/{project_id}")
async def get_data_for_energy_flows(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_df(models.EnergyFlow, user.id, project_id, db)
    df = df.reset_index(drop=True)
    return json.loads(df.to_json())


@app.get("/get_data_for_duration_curves/{project_id}")
async def get_data_for_duration_curves(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_df(models.DurationCurve, user.id, project_id, db)
    return json.loads(df.to_json())


@app.get("/get_co2_emissions_data/{project_id}")
async def get_co2_emissions_data(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    df = await queries.get_df(models.Emissions, user.id, project_id, db)
    return json.loads(df.to_json())


@app.post("/database_add_remove_automatic/{add_remove}/{project_id}")
async def database_add_remove_automatic(add_remove: str, project_id,
                                        selectBoundariesRequest: models.SelectBoundariesRequest,
                                        request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
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
        url = f'https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500]' \
              f'[bbox:{min_latitude},{min_longitude},{max_latitude},{max_longitude}];' \
              f'(way["building"="yes"];relation["building"];);out body;>;out skel qt;'
        url_formated = url.replace(" ", "+")
        with urllib.request.urlopen(url_formated) as url:
            data = json.loads(url.read().decode())

        # first converting the json file, which is delievered by overpass to geojson,
        # then obtaining coordinates and surface areas of all buildings inside the
        # 'big' rectangle.
        formated_geojson = bi.convert_overpass_json_to_geojson(data)
        (building_coord, building_area,) = bi.obtain_areas_and_mean_coordinates_from_geojson(formated_geojson)
        # excluding the buildings which are outside the drawn boundary
        features = formated_geojson["features"]
        mask_building_within_boundaries = {key: bi.is_point_in_boundaries(value, boundary_coordinates)
                                           for key, value in building_coord.items()}
        filtered_features = \
            [feature for feature in features if mask_building_within_boundaries[feature["property"]["@id"]]]
        formated_geojson["features"] = filtered_features
        building_coordidates_within_boundaries = \
            {key: value for key, value in building_coord.items() if mask_building_within_boundaries[key]}
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
        # Add the peak demand and average annual consumption for each node
        await demand_estimation(nodes=nodes, update_total_demand=False)
        # storing the nodes in the database
        await inserts.update_nodes_and_links(True, False, nodes, user.id, project_id, db)
    else:
        # reading the existing CSV file of nodes, and then removing the corresponding row
        # df = pd.read_csv(full_path_nodes)
        df = await queries.get_nodes_df(user.id, project_id, db)
        number_of_nodes = df.shape[0]
        for index in range(number_of_nodes):
            if bi.is_point_in_boundaries(point_coordinates=(df.to_dict()["latitude"][index],
                                                            df.to_dict()["longitude"][index],),
                                         boundaries=boundary_coordinates, ):
                df.drop(labels=index, axis=0, inplace=True)
        await inserts.update_nodes_and_links(True, False, df.to_dict(), user.id, project_id, db, False)


# add new manually-selected nodes to the *.csv file
@app.post("/database_add_remove_manual/{add_remove}/{project_id}")
async def database_add_remove_manual(add_remove: str, project_id, add_node_request: models.AddNodeRequest,
                                     request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    # headers = pd.read_csv(full_path_nodes).columns
    df = await queries.get_nodes_df(user.id, project_id, db)
    headers = df.columns
    nodes = {}
    nodes[headers[0]] = [add_node_request.latitude]
    nodes[headers[1]] = [add_node_request.longitude]
    nodes[headers[2]] = [add_node_request.node_type]
    nodes[headers[3]] = [add_node_request.consumer_type]
    nodes[headers[4]] = [add_node_request.consumer_detail]
    nodes[headers[5]] = [add_node_request.surface_area]
    nodes[headers[6]] = [add_node_request.peak_demand]
    nodes[headers[7]] = [add_node_request.average_consumption]
    nodes[headers[8]] = [add_node_request.is_connected]
    nodes[headers[9]] = [add_node_request.how_added]
    if add_remove == "remove":
        # reading the existing CSV file of nodes, and then removing the corresponding row
        # Since a node is removed, the calculated positions for ALL poles and
        # the power house must be first removed.
        df = df[(df["node_type"] != "pole") & (df["node_type"] != "power-house")]
        for index in df.index:
            if (round(add_node_request.latitude, 6) == df.to_dict()["latitude"][index]) and \
                    (round(add_node_request.longitude, 6) == df.to_dict()["longitude"][index]):
                df.drop(labels=index, axis=0, inplace=True)
        # storing the nodes in the database (updating the existing CSV file)
        df = df.reset_index(drop=True)
        await inserts.update_nodes_and_links(True, False, df.to_dict(), user.id, project_id, db, add=False)
    else:
        await inserts.update_nodes_and_links(True, False, nodes, user.id, project_id, db, add=True, replace=False)


def demand_estimation(nodes, update_total_demand):
    # after collecting all surface areas, based on a simple assumption, the peak demand will be obtained
    max_surface_area = max(nodes["surface_area"])

    # normalized demands is a CSV file with 5 columns representing the very low to very high demand profiles
    normalized_demands = pd.read_csv(full_path_demands, delimiter=";", header=None)

    if update_total_demand:
        # calculate the total peak demand for each of the five demand profiles to make the final demand profile
        peak_very_low_demand = 0
        peak_low_demand = 0
        peak_medium_demand = 0
        peak_high_demand = 0
        peak_very_high_demand = 0

        for area in nodes[nodes["is_connected"] == True]["surface_area"]:
            if area <= 0.2 * max_surface_area:
                peak_very_low_demand += 0.01 * area
            elif area < 0.4 * max_surface_area:
                peak_low_demand += 0.02 * area
            elif area < 0.6 * max_surface_area:
                peak_medium_demand += 0.03 * area
            elif area < 0.8 * max_surface_area:
                peak_high_demand += 0.04 * area
            else:
                peak_very_high_demand += 0.05 * area

        # create the total demand profile of the selected buildings
        total_demand = (
                normalized_demands.iloc[:, 0] * peak_very_low_demand
                + normalized_demands.iloc[:, 1] * peak_low_demand
                + normalized_demands.iloc[:, 2] * peak_medium_demand
                + normalized_demands.iloc[:, 3] * peak_high_demand
                + normalized_demands.iloc[:, 4] * peak_very_high_demand
        )

        # load timeseries data
        timeseries = pd.read_csv(full_path_timeseries)
        # replace the demand column in the timeseries file with the total demand calculated here
        timeseries["Demand"] = total_demand
        # update the CSV file
        timeseries.to_csv(full_path_timeseries, index=False)
    else:
        for area in nodes["surface_area"]:
            if area <= 0.2 * max_surface_area:
                nodes["peak_demand"].append(0.01 * area)
            elif area < 0.4 * max_surface_area:
                nodes["peak_demand"].append(0.02 * area)
            elif area < 0.6 * max_surface_area:
                nodes["peak_demand"].append(0.03 * area)
            elif area < 0.8 * max_surface_area:
                nodes["peak_demand"].append(0.04 * area)
            else:
                nodes["peak_demand"].append(0.05 * area)

        max_peak_demand = max(nodes["peak_demand"])
        counter = 0
        for peak_demand in nodes["peak_demand"]:
            if peak_demand <= 0.2 * max_peak_demand:
                nodes["average_consumption"].append(
                    normalized_demands.iloc[:, 0].sum() * nodes["peak_demand"][counter]
                )
            elif peak_demand < 0.4 * max_peak_demand:
                nodes["average_consumption"].append(
                    normalized_demands.iloc[:, 1].sum() * nodes["peak_demand"][counter]
                )
            elif peak_demand < 0.6 * max_peak_demand:
                nodes["average_consumption"].append(
                    normalized_demands.iloc[:, 2].sum() * nodes["peak_demand"][counter]
                )
            elif peak_demand < 0.8 * max_peak_demand:
                nodes["average_consumption"].append(
                    normalized_demands.iloc[:, 3].sum() * nodes["peak_demand"][counter]
                )
            else:
                nodes["average_consumption"].append(
                    normalized_demands.iloc[:, 4].sum() * nodes["peak_demand"][counter]
                )

            counter += 1

            # it is assumed that all nodes are parts of the mini-grid
            # later, when the shs candidates are obtained, the corresponding
            # values will be changed to 'False'
            nodes["is_connected"].append(True)

            # the node is selected automatically after drawing boundaries
            nodes["how_added"].append("automatic")

        return nodes


async def remove_rusults(user_id, project_id, db):
    await inserts.remove(models.Results, user_id, project_id, db)
    await inserts.remove(models.DemandCoverage, user_id, project_id, db)
    await inserts.remove(models.EnergyFlow, user_id, project_id, db)


@app.post("/optimize_grid/{project_id}")
async def optimize_grid(project_id, request: Request, db: Session = Depends(get_async_db)):
    user = await accounts.get_user_from_cookie(request, db)
    await remove_rusults(user.id, project_id, db)
    # Grab Currrent Time Before Running the Code
    start_execution_time = time.monotonic()

    # create GridOptimizer object
    df = await queries.get_input_df(user.id, project_id, db)

    opt = GridOptimizer(start_date=df.loc[0, "start_date"],
                        n_days=df.loc[0, "n_days"],
                        project_lifetime=df.loc[0, "project_lifetime"],
                        wacc=df.loc[0, "interest_rate"] / 100,
                        tax=0,)

    # get nodes from the database (CSV file) as a dictionary
    # then convert it again to a panda dataframe for simplicity
    # TODO: check the format of nodes from the database_read()
    nodes = await database_read(nodes_or_links="nodes", project_id=project_id, request=request, db=db)
    nodes = pd.DataFrame.from_dict(nodes)

    # if there is no element in the nodes, optimization will be terminated
    if len(nodes) == 0:
        return {"code": "success", "message": "Empty grid cannot be optimized!"}

    # initialite the database (remove contents of the CSV files)
    # otherwise, when clicking on the 'optimize' button, the existing system won't be removed

    # create a new "grid" object from the Grid class
    epc_distribution_cable = (
            (
                    opt.crf
                    * Optimizer.capex_multi_investment(
                opt,
                capex_0=df.loc[0, "distribution_cable_capex"],
                component_lifetime=df.loc[0, "distribution_cable_lifetime"],
            )
            )
            * opt.n_days
            / 365
    )

    epc_connection_cable = (
            (
                    opt.crf
                    * Optimizer.capex_multi_investment(
                opt,
                capex_0=df.loc[0, "connection_cable_capex"],
                component_lifetime=df.loc[0, "connection_cable_lifetime"],
            )
            )
            * opt.n_days
            / 365
    )

    epc_connection = (
            (
                    opt.crf
                    * Optimizer.capex_multi_investment(
                opt,
                capex_0=df.loc[0, "mg_connection_cost"],
                component_lifetime=opt.project_lifetime,
            )
            )
            * opt.n_days
            / 365
    )

    epc_pole = (
            (
                    opt.crf
                    * Optimizer.capex_multi_investment(
                opt,
                capex_0=df.loc[0, "pole_capex"],
                component_lifetime=df.loc[0, "pole_lifetime"],
            )
            )
            * opt.n_days
            / 365
    )

    # TODO: The following probability distribution function needs to be updated
    # considering the outcome of WP3

    # Assume the probability of each SHS tier level in the community.
    pdf_shs = [0.05, 0.1, 0.15, 0.4, 0.3]

    # This part calculated the total consumption of the community for the
    # selected time period.
    start_date_obj = opt.start_date
    start_datetime = datetime.combine(start_date_obj.date(), start_date_obj.time())
    end_datetime = start_datetime + timedelta(days=int(opt.n_days))

    # First, the demand for the entire year is read from the CSV file.
    demand_full_year = pd.read_csv(filepath_or_buffer=full_path_timeseries)
    demand_full_year.index = pd.date_range(
        start=start_datetime, periods=len(demand_full_year), freq="H"
    )

    # Then the demand for the selected time peroid given by the user will be
    # obtained.
    demand_selected_period = demand_full_year.Demand.loc[start_datetime:end_datetime]

    # The average consumption of the entire community in kWh for the selected
    # time period is calculated.
    average_consumption_selected_period = demand_selected_period.sum()

    # Total number of consumers that must be considered for calculating the
    # the total number of required SHS tier 1 to 3.
    n_consumers = nodes[nodes["node_type"] == "consumer"].shape[0]

    epc_shs = (
            (
                    opt.crf
                    * (
                        Optimizer.capex_multi_investment(
                            opt,
                            capex_0=(
                                            pdf_shs[0] * df.loc[0, "shs_tier_one_capex"]
                                            + pdf_shs[1] * df.loc[0, "shs_tier_two_capex"]
                                            + pdf_shs[2] * df.loc[0, "shs_tier_three_capex"]
                                    )
                                    * n_consumers
                                    + (
                                            pdf_shs[3] * df.loc[0, "shs_tier_four_capex"]
                                            + pdf_shs[4] * df.loc[0, "shs_tier_five_capex"]
                                    )
                                    * average_consumption_selected_period
                                    / 100,
                            component_lifetime=df.loc[0, "shs_lifetime"],
                        )
                    )
            )
            * opt.n_days
            / 365
    )

    grid = Grid(
        epc_distribution_cable=epc_distribution_cable,
        epc_connection_cable=epc_connection_cable,
        epc_connection=epc_connection,
        epc_pole=epc_pole,
        pole_max_connection=df.loc[0, "pole_max_n_connections"],
    )

    # make sure that the new grid object is empty before adding nodes to it
    grid.clear_nodes()
    grid.clear_all_links()

    # exclude solar-home-systems and poles from the grid optimization
    for node_index in nodes.index:
        if (
                (nodes.is_connected[node_index])
                and (not nodes.node_type[node_index] == "pole")
                and (not nodes.node_type[node_index] == "power-house")
        ):
            # add all consumers which are not served by solar-home-systems
            grid.add_node(
                label=str(node_index),
                longitude=nodes.longitude[node_index],
                latitude=nodes.latitude[node_index],
                node_type=nodes.node_type[node_index],
                is_connected=nodes.is_connected[node_index],
                peak_demand=nodes.peak_demand[node_index],
                average_consumption=nodes.average_consumption[node_index],
                surface_area=nodes.surface_area[node_index],
            )

    # convert all (long,lat) coordinates to (x,y) coordinates and update
    # the Grid object, which is necessary for the GridOptimizer
    grid.convert_lonlat_xy()

    # in case the grid contains 'poles' from the previous optimization
    # they must be removed, becasue the grid_optimizer will calculate
    # new locations for poles considering the newly added nodes
    grid.clear_poles()

    # Find the location of the power house which corresponds to the centroid
    # load of the village
    grid.get_load_centroid()

    # Calculate all distanced from the load centroid
    grid.get_nodes_distances_from_load_centroid()

    # Find the number of SHS consumers (temporarily)
    shs_share = 0
    n_total_consumers = grid.nodes.shape[0]
    n_shs_consumers = int(np.ceil(shs_share * n_total_consumers))
    n_mg_consumers = n_total_consumers - n_shs_consumers

    # Sort nodes based on their distance to the load center.
    grid.nodes.sort_values("distance_to_load_center", ascending=False, inplace=True)

    # Convert the first `n_shs_consumer` nodes into candidates for SHS.
    grid.nodes.loc[grid.nodes.index[0:n_shs_consumers], "is_connected"] = False

    # Sort nodes again based on their index label. Here, since the index is
    # string, sorting the nodes without changing the type of index would result
    # in a case, that '10' comes before '2'.
    grid.nodes.sort_index(key=lambda x: x.astype("int64"), inplace=True)

    # Create the demand profile for the energy system optimization based on the
    # number of mini-grid consumers.
    demand_estimation(nodes=grid.nodes, update_total_demand=True)

    # calculate the minimum number of poles based on the
    # maximum number of connectins at each pole
    if grid.pole_max_connection == 0:
        min_number_of_poles = 1
    else:
        min_number_of_poles = int(np.ceil(n_mg_consumers / (grid.pole_max_connection)))

    # ---------- MAX DISTANCE BETWEEN POLES AND CONSUMERS ----------
    connection_cable_max_length = df.loc[0, "connection_cable_max_length"]

    # First, the appropriate number of poles should be selected, to meet
    # the constraint on the maximum distance between consumers and poles.
    while True:
        # Initial number of poles.
        number_of_poles = opt.find_opt_number_of_poles(grid=grid, min_n_clusters=min_number_of_poles)

        # Find those connections with constraint violation.
        constraints_violation = grid.links[grid.links["link_type"] == "connection"]
        constraints_violation = constraints_violation[
            constraints_violation["length"] > connection_cable_max_length]

        # Increase the number of poles if necessary.
        if constraints_violation.shape[0] > 0:
            min_number_of_poles += 1
        else:
            break

    # ----------------- MAX DISTANCE BETWEEN POLES -----------------
    distribution_cable_max_length = df.loc[0, "distribution_cable_max_length"]

    # Find the connection links in the network with lengths greater than the
    # maximum allowed length for `connection` cables, specified by the user.
    long_links = grid.find_index_longest_distribution_link(
        max_distance_dist_links=distribution_cable_max_length,
    )

    # Add poles to the identified long `distribution` links, so that the
    # distance between all poles remains below the maximum allowed distance.
    grid.add_fixed_poles_on_long_links(
        long_links=long_links,
        max_allowed_distance=distribution_cable_max_length,
    )

    # Update the (lon,lat) coordinates based on the newly inserted poles
    # which only have (x,y) coordinates.
    grid.convert_lonlat_xy(inverse=True)

    # Connect all poles together using the minimum spanning tree algorithm.
    opt.connect_grid_poles(grid, long_links=long_links)

    # Calculate distances of all poles from the load centroid.
    grid.get_poles_distances_from_load_centroid()

    # Find the location of the power house.
    grid.select_location_of_power_house()

    # Calculate the cost of SHS.
    peak_demand_shs_consumers = grid.nodes[grid.nodes["is_connected"] == False].loc[
                                :, "peak_demand"
                                ]
    cost_shs = epc_shs * peak_demand_shs_consumers.sum()

    # get all poles obtained by the network relaxation method
    poles = grid.poles().reset_index(drop=True)

    # remove the unnecessary columns to make it compatible with the CSV files
    # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
    poles.drop(
        labels=[
            "x",
            "y",
            "cluster_label",
            "type_fixed",
            "n_connection_links",
            "n_distribution_links",
        ],
        axis=1,
        inplace=True,
    )

    # Store the list of poles in the "node" database.
    await inserts.update_nodes_and_links(True, False, poles.to_dict(), user.id, project_id, db)

    # get all links obtained by the network relaxation method
    links = grid.links.reset_index(drop=True)

    # remove the unnecessary columns to make it compatible with the CSV files
    # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
    links.drop(
        labels=[
            "x_from",
            "y_from",
            "x_to",
            "y_to",
            "n_consumers",
            "total_power",
            "from_node",
            "to_node",
        ],
        axis=1,
        inplace=True,
    )

    # store the list of poles in the "node" database
    await inserts.update_nodes_and_links(False, True, links.to_dict(), user.id, project_id, db)

    # Grab Currrent Time After Running the Code
    end_execution_time = time.monotonic()

    # store data for showing in the final results
    df = await queries.get_results_df(user.id, project_id, db)
    df.loc[0, "n_consumers"] = len(grid.consumers())
    df.loc[0, "n_shs_consumers"] = n_shs_consumers
    df.loc[0, "n_poles"] = len(grid.poles())
    df.loc[0, "length_distribution_cable"] = int(
        grid.links[grid.links.link_type == "distribution"]["length"].sum()
    )
    df.loc[0, "length_connection_cable"] = int(
        grid.links[grid.links.link_type == "connection"]["length"].sum()
    )
    df.loc[0, "cost_grid"] = int(grid.cost())
    df.loc[0, "cost_shs"] = int(cost_shs)
    df.loc[0, "time_grid_design"] = end_execution_time - start_execution_time
    df.loc[0, "n_distribution_links"] = int(
        grid.links[grid.links["link_type"] == "distribution"].shape[0]
    )
    df.loc[0, "n_connection_links"] = int(
        grid.links[grid.links["link_type"] == "connection"].shape[0]
    )

    inserts.insert_results_df(df, user.id, project_id, db)

    grid.find_n_links_connected_to_each_pole()

    grid.find_capacity_of_each_link()

    grid.distribute_grid_cost_among_consumers()


@app.post("/optimize_energy_system/{project_id}")
async def optimize_energy_system(project_id, request: Request, optimize_energy_system_request:
models.OptimizeEnergySystemRequest, db: Session = Depends(get_db)):
    user = await accounts.get_user_from_cookie(request, db)
    # Grab Currrent Time Before Running the Code
    start_execution_time = time.monotonic()

    df = await queries.get_input_df(user.id, project_id, db)

    solver = 'gurobi' if po.SolverFactory('gurobi').available() else 'cbc'

    ensys_opt = EnergySystemOptimizer(
        start_date=df.loc[0, "start_date"],
        n_days=df.loc[0, "n_days"],
        project_lifetime=df.loc[0, "project_lifetime"],
        wacc=df.loc[0, "interest_rate"] / 100,
        tax=0,
        path_data=full_path_timeseries,
        solver=solver,
        pv=optimize_energy_system_request.pv,
        diesel_genset=optimize_energy_system_request.diesel_genset,
        battery=optimize_energy_system_request.battery,
        inverter=optimize_energy_system_request.inverter,
        rectifier=optimize_energy_system_request.rectifier,
        shortage=optimize_energy_system_request.shortage,
    )
    ensys_opt.optimize_energy_system()

    # Grab Currrent Time After Running the Code
    end_execution_time = time.monotonic()

    # unit for co2_emission_factor is kgCO2 per kWh of produced electricity
    if ensys_opt.capacity_genset < 60:
        co2_emission_factor = 1.580
    elif ensys_opt.capacity_genset < 300:
        co2_emission_factor = 0.883
    else:
        co2_emission_factor = 0.699

    # store fuel co2 emissions (kg_CO2 per L of fuel)
    df = pd.DataFrame()
    df["non_renewable_electricity_production"] = (
            np.cumsum(ensys_opt.demand) * co2_emission_factor / 1000
    )  # tCO2 per year
    df["hybrid_electricity_production"] \
        = np.cumsum(ensys_opt.sequences_genset) * co2_emission_factor / 1000  # tCO2 per year
    df["co2_savings"] = \
        df.loc[:, "non_renewable_electricity_production"] - df.loc[:, "hybrid_electricity_production"]  # tCO2 per year
    df['h'] = np.arange(1, len(ensys_opt.demand) + 1)
    inserts.insert_df(models.Emissions, df, user.id, project_id, db)
    # TODO: -2 must actually be -1, but for some reason, the co2-emission csv file has an additional empty row
    co2_savings = df.loc[:, "co2_savings"][
        -2
    ]  # takes the last element of the cumulative sum

    # store data for showing in the final results
    df = await queries.get_results_df(user.id, project_id, db)
    df.loc[0, "cost_renewable_assets"] = ensys_opt.total_renewable
    df.loc[0, "cost_non_renewable_assets"] = ensys_opt.total_non_renewable
    df.loc[0, "cost_fuel"] = ensys_opt.total_fuel
    df.loc[0, "lcoe"] = (
            100
            * (ensys_opt.total_revenue + df.loc[0, "cost_grid"] + df.loc[0, "cost_shs"])
            / ensys_opt.total_demand
    )
    df.loc[0, "res"] = ensys_opt.res
    df.loc[0, "shortage_total"] = ensys_opt.shortage
    df.loc[0, "surplus_rate"] = ensys_opt.surplus_rate
    df.loc[0, "pv_capacity"] = ensys_opt.capacity_pv
    df.loc[0, "battery_capacity"] = ensys_opt.capacity_battery
    df.loc[0, "inverter_capacity"] = ensys_opt.capacity_inverter
    df.loc[0, "rectifier_capacity"] = ensys_opt.capacity_rectifier
    df.loc[0, "diesel_genset_capacity"] = ensys_opt.capacity_genset
    df.loc[0, "peak_demand"] = ensys_opt.demand_peak
    df.loc[0, "surplus"] = ensys_opt.sequences_surplus.max()
    # data for sankey diagram - all in MWh
    df.loc[0, "fuel_to_diesel_genset"] = (
            ensys_opt.sequences_fuel_consumption.sum()
            * 0.846
            * ensys_opt.diesel_genset["parameters"]["fuel_lhv"]
            / 1000
    )
    df.loc[0, "diesel_genset_to_rectifier"] = (
            ensys_opt.sequences_rectifier.sum()
            / ensys_opt.rectifier["parameters"]["efficiency"]
            / 1000
    )
    df.loc[0, "diesel_genset_to_demand"] = (
            ensys_opt.sequences_genset.sum() / 1000
            - df.loc[0, "diesel_genset_to_rectifier"]
    )
    df.loc[0, "rectifier_to_dc_bus"] = ensys_opt.sequences_rectifier.sum() / 1000
    df.loc[0, "pv_to_dc_bus"] = ensys_opt.sequences_pv.sum() / 1000
    df.loc[0, "battery_to_dc_bus"] = ensys_opt.sequences_battery_discharge.sum() / 1000
    df.loc[0, "dc_bus_to_battery"] = ensys_opt.sequences_battery_charge.sum() / 1000
    df.loc[0, "dc_bus_to_inverter"] = (
            ensys_opt.sequences_inverter.sum()
            / ensys_opt.inverter["parameters"]["efficiency"]
            / 1000
    )
    df.loc[0, "dc_bus_to_surplus"] = ensys_opt.sequences_surplus.sum() / 1000
    df.loc[0, "inverter_to_demand"] = ensys_opt.sequences_inverter.sum() / 1000
    df.loc[0, "time_energy_system_design"] = end_execution_time - start_execution_time
    df.loc[0, "co2_savings"] = co2_savings
    inserts.insert_results_df(df, user.id, project_id, db)

    # store energy flows
    df = pd.DataFrame()
    df["diesel_genset_production"] = ensys_opt.sequences_genset
    df["pv_production"] = ensys_opt.sequences_pv
    df["battery_charge"] = ensys_opt.sequences_battery_charge
    df["battery_discharge"] = ensys_opt.sequences_battery_discharge
    df["battery_content"] = ensys_opt.sequences_battery_content
    df["demand"] = ensys_opt.sequences_demand
    df["surplus"] = ensys_opt.sequences_surplus
    inserts.insert_df(models.EnergyFlow, df, user.id, project_id, db)

    df = pd.DataFrame()
    df["demand"] = ensys_opt.sequences_demand
    df["renewable"] = ensys_opt.sequences_inverter
    df["non_renewable"] = ensys_opt.sequences_genset
    df["surplus"] = ensys_opt.sequences_surplus
    df.index.name = "dt"
    df = df.reset_index()
    inserts.insert_demand_coverage_df(df, user.id, project_id, db)

    # store duration curves
    df = pd.DataFrame()
    df["diesel_genset_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_genset) + 1)
                                      / len(ensys_opt.sequences_genset))
    df["diesel_genset_duration"] = (100 * np.sort(ensys_opt.sequences_genset)[::-1]
                                    / ensys_opt.sequences_genset.max())
    df["pv_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_pv) + 1)
                           / len(ensys_opt.sequences_pv))
    df["pv_duration"] = (
            100 * np.sort(ensys_opt.sequences_pv)[::-1] / ensys_opt.sequences_pv.max())
    df["rectifier_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_rectifier) + 1)
                                  / len(ensys_opt.sequences_rectifier))
    df["rectifier_duration"] = 100 * np.nan_to_num(
        np.sort(ensys_opt.sequences_rectifier)[::-1]
        / ensys_opt.sequences_rectifier.max())
    df["inverter_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_inverter) + 1)
                                 / len(ensys_opt.sequences_inverter))
    df["inverter_duration"] = (100 * np.sort(ensys_opt.sequences_inverter)[::-1]
                               / ensys_opt.sequences_inverter.max())
    df["battery_charge_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_battery_charge) + 1)
                                       / len(ensys_opt.sequences_battery_charge))
    df["battery_charge_duration"] = (100 * np.sort(ensys_opt.sequences_battery_charge)[::-1]
                                     / ensys_opt.sequences_battery_charge.max())
    df["battery_discharge_percentage"] = (100 * np.arange(1, len(ensys_opt.sequences_battery_discharge) + 1)
                                          / len(ensys_opt.sequences_battery_discharge))
    df["battery_discharge_duration"] = (100 * np.sort(ensys_opt.sequences_battery_discharge)[::-1]
                                        / ensys_opt.sequences_battery_discharge.max())
    df['h'] = np.arange(1, len(ensys_opt.sequences_genset) + 1)
    inserts.insert_df(models.DurationCurve, df, user.id, project_id, db)


@app.post("/shs_identification/")
def identify_shs(shs_identification_request: models.ShsIdentificationRequest):
    print("starting shs_identification...")
    # res = db.execute("select * from nodes")
    # nodes = res.fetchall()
    if len(nodes) == 0:
        return {"code": "success", "message": "No nodes in table, no identification to be performed", }
    # use latitude of the node that is the most west to set origin of x coordinates
    ref_latitude = min([node[1] for node in nodes])
    # use latitude of the node that is the most south to set origin of y coordinates
    ref_longitude = min([node[2] for node in nodes])
    nodes_df = shs_ident.create_nodes_df()
    cable_price_per_meter = (shs_identification_request.cable_price_per_meter_for_shs_mst_identification)
    additional_price_for_connection_per_node = (shs_identification_request.connection_cost_to_minigrid)
    for node in nodes:
        latitude = math.radians(node[1])
        longitude = math.radians(node[2])
        x, y = conv.xy_coordinates_from_latitude_longitude(
            latitude=latitude,
            longitude=longitude,
            ref_latitude=ref_latitude,
            ref_longitude=ref_longitude,
        )
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

        shs_ident.add_node(
            nodes_df,
            node_label,
            x,
            y,
            required_capacity,
            max_power,
            shs_price=shs_price,
        )
    links_df = shs_ident.mst_links(nodes_df)
    start_time = time.time()
    if shs_identification_request.algo == "mst1":
        nodes_to_disconnect_from_grid = shs_ident.nodes_to_disconnect_from_grid(
            nodes_df=nodes_df,
            links_df=links_df,
            cable_price_per_meter=cable_price_per_meter,
            additional_price_for_connection_per_node=additional_price_for_connection_per_node,
        )
        print(
            f"execution time for shs identification (mst1): {time.time() - start_time} s"
        )
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
