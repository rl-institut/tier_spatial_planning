import datetime
import fastapi_app.tools.boundary_identification as bi
import fastapi_app.tools.coordinates_conversion as conv
import fastapi_app.tools.shs_identification as shs_ident
import fastapi_app.db.models as models
from fastapi import FastAPI, Request, APIRouter, Response
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi_app.db.database import engine
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.optimizer import Optimizer, GridOptimizer, EnergySystemOptimizer, po
from fastapi import Depends
from sqlalchemy.orm import Session
from fastapi_app.tools.accounts import Hasher, create_guid, is_valid_credentials, send_activation_link, activate_mail, \
    authenticate_user, create_access_token
from fastapi_app.tools import accounts
from fastapi_app.db import config
from fastapi_app.db.database import get_db
import math
import urllib.request
import ssl
import json
import pandas as pd
import numpy as np
import os

# for debugging
import uvicorn

# for appending to the dictionary
from collections import defaultdict

# for sending an array of data from JS to the fastAPI
from typing import Any, Dict, List, Union

# import the builtin time module
import time

app = FastAPI()

app.mount(
    "/fastapi_app/static", StaticFiles(directory="fastapi_app/static"), name="static"
)

models.Base.metadata.create_all(bind=engine)

templates = Jinja2Templates(directory="fastapi_app/pages")

# define different directories for:
# (1) database: *.csv files for nodes and links,
# (2) inputs: input excel files (cost data and timeseries) for offgridders + web app import and export files, and
# (3) outputs: offgridders results
directory_parent = "fastapi_app"

directory_database = os.path.join(directory_parent, "data", "database").replace(
    "\\", "/"
)
full_path_nodes = os.path.join(directory_database, "nodes.csv").replace("\\", "/")
full_path_links = os.path.join(directory_database, "links.csv").replace("\\", "/")
full_path_demands = os.path.join(directory_database, "demands.csv").replace("\\", "/")
full_path_stored_inputs = os.path.join(directory_database, "stored_inputs.csv").replace(
    "\\", "/"
)
full_path_stored_results = os.path.join(
    directory_database, "stored_results.csv"
).replace("\\", "/")
full_path_demand_coverage = os.path.join(
    directory_database, "demand_coverage.csv"
).replace("\\", "/")
full_path_energy_flows = os.path.join(directory_database, "energy_flows.csv").replace(
    "\\", "/"
)
full_path_duration_curves = os.path.join(
    directory_database, "duration_curves.csv"
).replace("\\", "/")
full_path_co2_emissions = os.path.join(directory_database, "co2_emissions.csv").replace(
    "\\", "/"
)
os.makedirs(directory_database, exist_ok=True)

directory_inputs = os.path.join(directory_parent, "data", "inputs").replace("\\", "/")
full_path_timeseries = os.path.join(directory_inputs, "timeseries.csv").replace(
    "\\", "/"
)
os.makedirs(directory_inputs, exist_ok=True)

# this is to avoid problems in "urllib" by not authenticating SSL certificate, otherwise following error occurs:
# urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1131)>
ssl._create_default_https_context = ssl._create_unverified_context

# define the template for importing json data in the form of arrays from js to python
json_object = Dict[Any, Any]
json_array = List[Any]
import_structure = Union[json_array, json_object]

"""
api_router = APIRouter()
api_router.include_router(route_users.router, prefix="/users", tags=["users"])
app.include_router(api_router)
"""


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


@app.get(
    "/download_export_file",
    responses={
        200: {
            "description": "xlsx file containing the information about the configuration.",
            "content": {
                "static/io/test_excel_node.xlsx": {"example": "No example available."}
            },
        }
    },
)
async def download_export_file():
    file_name = "temp.xlsx"
    # Download xlsx file
    file_path = os.path.join(directory_parent, f"import_export/{file_name}")

    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="backup.xlsx",
        )
    else:
        return {"error": "File not found!"}


@app.post("/import_data")
async def import_data(import_files: import_structure = None):
    # add nodes from the 'nodes' sheet of the excel file to the 'nodes.csv' file
    # TODO: update the template for adding nodes
    nodes = import_files["nodes_to_import"]
    links = import_files["links_to_import"]
    if len(nodes) > 0:
        database_add(add_nodes=True, add_links=False, inlet=nodes)

    if len(links) > 0:
        database_add(add_nodes=False, add_links=True, inlet=links)

    # ------------------------------ HANDLE REQUEST ------------------------------#


@app.get("/")
async def home(request: Request):
    if os.path.exists(full_path_stored_inputs) is False:
        header_stored_inputs = [
            "project_name",
            "project_description",
            "interest_rate",
            "project_lifetime",
            "start_date",
            "temporal_resolution",
            "n_days",
            "distribution_cable_lifetime",
            "distribution_cable_capex",
            "distribution_cable_max_length",
            "connection_cable_lifetime",
            "connection_cable_capex",
            "connection_cable_max_length",
            "pole_lifetime",
            "pole_capex",
            "pole_max_n_connections",
            "mg_connection_cost",
            "mg_n_operators",
            "mg_salary_operator",
            "shs_lifetime",
            "shs_tier_one_capex",
            "shs_tier_two_capex",
            "shs_tier_three_capex",
            "shs_tier_four_capex",
            "shs_tier_five_capex",
        ]
        pd.DataFrame(columns=header_stored_inputs).to_csv(
            full_path_stored_inputs, index=False
        )

    # return templates.TemplateResponse("project-setup.html", {"request": request})
    return templates.TemplateResponse("landing-page.html", {"request": request})


@app.get("/project_setup")
async def project_setup(request: Request):
    return templates.TemplateResponse("project-setup.html", {"request": request})


@app.get("/navbar")
async def project_setup(request: Request):
    return templates.TemplateResponse("_navbar.html", {"request": request})


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
    user = accounts.get_user_from_cookie(request, db)
    if user is None:
        return templates.TemplateResponse("landing-page.html", {"request": request})
    else:
        return templates.TemplateResponse("account_overview.html", {"request": request})


@app.get("/consumer_selection")
async def consumer_selection(request: Request):
    return templates.TemplateResponse("consumer-selection.html", {"request": request})


@app.get("/grid_design")
async def grid_design(request: Request):
    return templates.TemplateResponse("grid-design.html", {"request": request})


@app.get("/energy_system_design")
async def energy_system_design(request: Request):
    return templates.TemplateResponse("energy-system-design.html", {"request": request})


@app.get("/simulation_results")
async def simulation_results(request: Request):
    return templates.TemplateResponse("simulation-results.html", {"request": request})


@app.get("/get_demand_coverage_data/")
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
        "how_added",
    ]
    header_links = ["lat_from", "lon_from", "lat_to", "lon_to", "link_type", "length"]
    header_stored_results = [
        "n_consumers",
        "n_shs_consumers",
        "n_poles",
        "n_distribution_links",
        "n_connection_links",
        "length_distribution_cable",
        "average_length_distribution_cable",
        "length_connection_cable",
        "average_length_connection_cable",
        "cost_grid",
        "cost_shs",
        "lcoe",
        "res",
        "shortage_total",
        "surplus_rate",
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
        "time_grid_design",
        "time_energy_system_design",
        "time",
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
    header_demand_coverage = ["demand", "renewable", "non-renewable", "surplus"]
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

    pd.DataFrame(columns=header_stored_results).to_csv(
        full_path_stored_results, index=False
    )
    pd.DataFrame(columns=header_energy_flows).to_csv(
        full_path_energy_flows, index=False
    )
    pd.DataFrame(columns=header_duration_curves).to_csv(
        full_path_duration_curves, index=False
    )
    pd.DataFrame(columns=header_co2_emissions).to_csv(
        full_path_co2_emissions, index=False
    )
    pd.DataFrame(columns=header_demand_coverage).to_csv(
        full_path_demand_coverage, index=False
    )


# add new manually-selected nodes to the *.csv file
@app.post("/database_add_remove_manual/{add_remove}")
async def database_add_remove_manual(
        add_remove: str, add_node_request: models.AddNodeRequest
):
    headers = pd.read_csv(full_path_nodes).columns
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
        df = pd.read_csv(full_path_nodes)

        # Since a node is removed, the calculated positions for ALL poles and
        # the power house must be first removed.
        df = df[(df["node_type"] != "pole") & (df["node_type"] != "power-house")]

        for index in df.index:
            if (
                    round(add_node_request.latitude, 6) == df.to_dict()["latitude"][index]
            ) and (
                    round(add_node_request.longitude, 6) == df.to_dict()["longitude"][index]
            ):
                df.drop(labels=index, axis=0, inplace=True)

        # Remove all existing nodes and links.
        await database_initialization(nodes=True, links=True)

        # storing the nodes in the database (updating the existing CSV file)
        df = df.reset_index(drop=True)
        database_add(add_nodes=True, add_links=False, inlet=df.to_dict())
    else:
        database_add(add_nodes=True, add_links=False, inlet=nodes)


# add new nodes/links to the database
def database_add(add_nodes: bool, add_links: bool, inlet: dict):
    # updating csv files based on the added nodes
    if add_nodes:
        nodes = inlet
        # newly added nodes
        df = pd.DataFrame.from_dict(nodes).round(decimals=6)

        # the existing database
        if os.path.exists(full_path_nodes):
            df_existing = pd.read_csv(full_path_nodes)

            # Remove all existing poles and the power house.
            df_existing = df_existing[
                (df_existing["node_type"] != "pole")
                & (df_existing["node_type"] != "power-house")
                ]
        else:
            df_existing = pd.DataFrame()

            # Aappend the existing database with the new nodes and remove
        # duplicates (only when both lat and lon are identical).
        df_total = df_existing.append(df).drop_duplicates(
            subset=["latitude", "longitude", "node_type"], inplace=False
        )

        # storing the nodes in the database (updating the existing CSV file).
        df_total = df_total.reset_index(drop=True)

        # If consumers are added to the database or removed from it, all
        # already existing links and all existing poles must be removed.
        if df["node_type"].str.contains("consumer").sum() > 0:
            # Remove existing links.
            if os.path.exists(full_path_links):
                df_links = pd.read_csv(full_path_links)
                df_links.drop(labels=df_links.index, axis=0, inplace=True)
                df_links.to_csv(full_path_links, index=False, header=df_links.head)

        # defining the precision of data
        df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
        df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
        df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
        df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
        df_total.average_consumption = df_total.average_consumption.map(
            lambda x: "%.3f" % x
        )

        # finally adding the refined dataframe (if it is not empty) to the existing csv file
        if len(df_total.index) != 0:
            df_total.to_csv(full_path_nodes, index=False, header=df_existing.head)

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
            df.to_csv(
                full_path_links,
                mode="a",
                header=False,
                index=False,
                float_format="%.0f",
            )


@app.get("/database_to_js/{nodes_or_links}")
async def database_read(nodes_or_links: str):
    # importing nodes and links from the csv files to the map
    if nodes_or_links == "nodes":
        nodes_list = json.loads(pd.read_csv(full_path_nodes).to_json())
        return nodes_list
    else:
        links_list = json.loads(pd.read_csv(full_path_links).to_json())
        return links_list


@app.get("/load_results/")
async def load_results():
    df = pd.read_csv(full_path_stored_results)
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
async def load_previous_data(page_name):
    df = pd.read_csv(full_path_stored_inputs)
    # In case the CSV file containing all stored inputs is empty, the following
    # conditions will not be executed.
    if not df.empty:
        if page_name == "project_setup":
            selection = ['project_name', 'project_description', 'interest_rate', 'project_lifetime',
                         'start_date', 'temporal_resolution', 'n_days']
        elif page_name == "grid_design":
            selection = ['distribution_cable_lifetime', 'distribution_cable_capex', 'distribution_cable_max_length',
                         'connection_cable_lifetime', 'connection_cable_capex', 'connection_cable_max_length',
                         'pole_lifetime', 'pole_capex', 'pole_max_n_connections', 'mg_connection_cost',
                         'mg_n_operators', 'mg_salary_operator', 'shs_lifetime', 'shs_tier_one_capex',
                         'shs_tier_two_capex', 'shs_tier_three_capex', 'shs_tier_four_capex', 'shs_tier_five_capex']
        df = df[selection].astype(str)
    previous_data = df.to_dict(orient='records')[0]
    # importing nodes and links from the csv files to the map
    return previous_data


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


@app.post("/get_access_token/", response_model=models.Token)
def get_access_token(response: Response, credentials: models.Credentials, db: Session = Depends(get_db)):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        user = authenticate_user(credentials.email, credentials.password, db)
        name = user.email
    else:
        name = 'anonymous'
    access_token_expires = datetime.timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": name}, expires_delta=access_token_expires)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}",
                        httponly=True)  # set HttpOnly cookie in response
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/logout/")
async def logout(response: Response):
    response.delete_cookie("access_token")


@app.post("/query_account_data/")
async def query_account_data(request: Request, db: Session = Depends(get_db)):
    user = accounts.get_user_from_cookie(request, db)
    return models.UserOverview(email=user.email)


@app.post("/save_previous_data/{page_name}")
async def save_previous_data(
        page_name: str, save_previous_data_request: models.SavePreviousDataRequest
):
    df = pd.read_csv(full_path_stored_inputs)

    if page_name == "project_setup":
        selection = ['project_name', 'project_description', 'interest_rate', 'project_lifetime',
                     'start_date', 'temporal_resolution', 'n_days']
        for col in selection:
            df.loc[0, col] = save_previous_data_request.page_setup[col]
    elif page_name == "grid_design":
        selection = ['distribution_cable_lifetime', 'distribution_cable_capex', 'distribution_cable_max_length',
                     'connection_cable_lifetime', 'connection_cable_capex', 'connection_cable_max_length',
                     'pole_lifetime', 'pole_capex', 'pole_max_n_connections', 'mg_connection_cost',
                     'mg_n_operators', 'mg_salary_operator', 'shs_lifetime', 'shs_tier_one_capex',
                     'shs_tier_two_capex', 'shs_tier_three_capex', 'shs_tier_four_capex', 'shs_tier_five_capex']
        for col in selection:
            df.loc[0, col] = save_previous_data_request.grid_design[col]

    # save the updated dataframe
    df.to_csv(full_path_stored_inputs, index=False)


@app.get("/get_optimal_capacities/")
async def get_optimal_capacities():
    df = pd.read_csv(full_path_stored_results)
    df = df.rename(columns={'pv_capacity': 'pv', 'battery_capacity': 'battery', 'inverter_capacity': 'inverter',
                            'rectifier_capacity': 'rectifier', 'diesel_genset_capacity': 'diesel_genset'})
    df = df[['pv', 'battery', 'inverter', 'rectifier', 'diesel_genset', 'peak_demand', 'surplus']].astype(str)
    optimal_capacities = df.to_dict(orient='records')[0]

    # importing nodes and links from the csv files to the map
    return optimal_capacities


@app.get("/get_lcoe_breakdown/")
async def get_lcoe_breakdown():
    df = pd.read_csv(full_path_stored_results)
    df = df.rename(columns={'cost_renewable_assets': 'renewable_assets',
                            'cost_non_renewable_assets': 'non_renewable_assets',
                            'cost_grid': 'grid',
                            'cost_fuel': 'fuel'})
    df = df[['renewable_assets', 'non_renewable_assets', 'grid', 'fuel']].astype(str)
    lcoe_breakdown = df.to_dict(orient='records')[0]
    # importing nodes and links from the csv files to the map
    return lcoe_breakdown


@app.get("/get_data_for_sankey_diagram/")
async def get_data_for_sankey_diagram():
    df = pd.read_csv(full_path_stored_results)
    df = df[['fuel_to_diesel_genset', 'diesel_genset_to_rectifier', 'diesel_genset_to_demand',
             'rectifier_to_dc_bus', 'pv_to_dc_bus', 'battery_to_dc_bus', 'dc_bus_to_battery', 'dc_bus_to_inverter',
             'dc_bus_to_surplus', 'inverter_to_demand']]
    df = df.astype(str)
    sankey_data = df.to_dict(orient='records')[0]

    # importing nodes and links from the csv files to the map
    return sankey_data


@app.get("/get_data_for_energy_flows/")
async def get_data_for_energy_flows():
    return json.loads(pd.read_csv(full_path_energy_flows).to_json())


@app.get("/get_data_for_duration_curves/")
async def get_data_for_duration_curves():
    return json.loads(pd.read_csv(full_path_duration_curves).to_json())


@app.get("/get_co2_emissions_data/")
async def get_co2_emissions_data():
    return json.loads(pd.read_csv(full_path_co2_emissions).to_json())


@app.post("/database_add_remove_automatic/{add_remove}")
async def database_add_remove_automatic(
        add_remove: str, selectBoundariesRequest: models.SelectBoundariesRequest
):
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
        (
            building_coord,
            building_area,
        ) = bi.obtain_areas_and_mean_coordinates_from_geojson(formated_geojson)

        # excluding the buildings which are outside the drawn boundary
        features = formated_geojson["features"]
        mask_building_within_boundaries = {
            key: bi.is_point_in_boundaries(value, boundary_coordinates)
            for key, value in building_coord.items()
        }
        filtered_features = [
            feature
            for feature in features
            if mask_building_within_boundaries[feature["property"]["@id"]]
        ]
        formated_geojson["features"] = filtered_features
        building_coordidates_within_boundaries = {
            key: value
            for key, value in building_coord.items()
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

        # Add the peak demand and average annual consumption for each node
        demand_estimation(nodes=nodes, update_total_demand=False)

        # storing the nodes in the database
        database_add(add_nodes=True, add_links=False, inlet=nodes)

    else:
        # reading the existing CSV file of nodes, and then removing the corresponding row
        df = pd.read_csv(full_path_nodes)
        number_of_nodes = df.shape[0]
        for index in range(number_of_nodes):
            if bi.is_point_in_boundaries(
                    point_coordinates=(
                            df.to_dict()["latitude"][index],
                            df.to_dict()["longitude"][index],
                    ),
                    boundaries=boundary_coordinates,
            ):
                df.drop(labels=index, axis=0, inplace=True)

        # removing all nodes and links
        await database_initialization(nodes=True, links=True)

        # storing the nodes in the database (updating the existing CSV file)
        df = df.reset_index(drop=True)
        database_add(add_nodes=True, add_links=False, inlet=df.to_dict())


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


@app.post("/optimize_grid/")
async def optimize_grid():
    # Grab Currrent Time Before Running the Code
    start_execution_time = time.monotonic()

    # create GridOptimizer object
    df = pd.read_csv(full_path_stored_inputs)

    opt = GridOptimizer(
        start_date=df.loc[0, "start_date"],
        n_days=df.loc[0, "n_days"],
        project_lifetime=df.loc[0, "project_lifetime"],
        wacc=df.loc[0, "interest_rate"] / 100,
        tax=0,
    )

    # get nodes from the database (CSV file) as a dictionary
    # then convert it again to a panda dataframe for simplicity
    # TODO: check the format of nodes from the database_read()
    nodes = await database_read(nodes_or_links="nodes")
    nodes = pd.DataFrame.from_dict(nodes)

    # if there is no element in the nodes, optimization will be terminated
    if len(nodes) == 0:
        return {"code": "success", "message": "Empty grid cannot be optimized!"}

    # initialite the database (remove contents of the CSV files)
    # otherwise, when clicking on the 'optimize' button, the existing system won't be removed
    await database_initialization(nodes=False, links=True)

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

    epc_shs = (
            (
                    opt.crf
                    * (
                        Optimizer.capex_multi_investment(
                            opt,
                            capex_0=df.loc[0, "shs_tier_one_capex"]
                                    + df.loc[0, "shs_tier_two_capex"]
                                    + df.loc[0, "shs_tier_three_capex"],
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
    grid.nodes.at[grid.nodes.index[0:n_shs_consumers], "is_connected"] = False

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
        number_of_poles = opt.find_opt_number_of_poles(
            grid=grid, min_n_clusters=min_number_of_poles
        )

        # Find those connections with constraint violation.
        constraints_violation = grid.links[grid.links["link_type"] == "connection"]
        constraints_violation = constraints_violation[
            constraints_violation["length"] > connection_cable_max_length
            ]

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
            "segment",
            "type_fixed",
            "allocation_capacity",
        ],
        axis=1,
        inplace=True,
    )

    # Store the list of poles in the "node" database.
    database_add(add_nodes=True, add_links=False, inlet=poles.to_dict())

    # get all links obtained by the network relaxation method
    links = grid.links.reset_index(drop=True)

    # remove the unnecessary columns to make it compatible with the CSV files
    # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
    links.drop(labels=["x_from", "y_from", "x_to", "y_to"], axis=1, inplace=True)

    # store the list of poles in the "node" database
    database_add(add_nodes=False, add_links=True, inlet=links.to_dict())

    # Grab Currrent Time After Running the Code
    end_execution_time = time.monotonic()

    # store data for showing in the final results
    df = pd.read_csv(full_path_stored_results)
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

    df.to_csv(
        full_path_stored_results,
        mode="a",
        header=False,
        index=False,
        float_format="%.0f",
    )


@app.post("/optimize_energy_system/")
async def optimize_energy_system(
        optimize_energy_system_request: models.OptimizeEnergySystemRequest,
):
    # Grab Currrent Time Before Running the Code
    start_execution_time = time.monotonic()

    df = pd.read_csv(full_path_stored_inputs)

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
    df = pd.read_csv(full_path_co2_emissions)
    df.loc[:, "non_renewable_electricity_production"] = (
            np.cumsum(ensys_opt.demand) * co2_emission_factor / 1000
    )  # tCO2 per year
    df.loc[:, "hybrid_electricity_production"] = (
            np.cumsum(ensys_opt.sequences_genset) * co2_emission_factor / 1000
    )  # tCO2 per year
    df.loc[:, "co2_savings"] = (
            df.loc[:, "non_renewable_electricity_production"]
            - df.loc[:, "hybrid_electricity_production"]
    )  # tCO2 per year
    df.to_csv(full_path_co2_emissions, index=False, float_format="%.3f")
    # TODO: -2 must actually be -1, but for some reason, the co2-emission csv file has an additional empty row
    co2_savings = df.loc[:, "co2_savings"][
        -2
    ]  # takes the last element of the cumulative sum

    # store data for showing in the final results
    df = pd.read_csv(full_path_stored_results)
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
    df.to_csv(full_path_stored_results, index=False, float_format="%.1f")

    # store energy flows
    df = pd.read_csv(full_path_energy_flows)
    df.loc[:, "diesel_genset_production"] = ensys_opt.sequences_genset
    df.loc[:, "pv_production"] = ensys_opt.sequences_pv
    df.loc[:, "battery_charge"] = ensys_opt.sequences_battery_charge
    df.loc[:, "battery_discharge"] = ensys_opt.sequences_battery_discharge
    df.loc[:, "battery_content"] = ensys_opt.sequences_battery_content
    df.loc[:, "demand"] = ensys_opt.sequences_demand
    df.loc[:, "surplus"] = ensys_opt.sequences_surplus
    df.to_csv(
        full_path_energy_flows, index=True, index_label="time", float_format="%.3f"
    )

    # store demand coverage
    df = pd.read_csv(full_path_demand_coverage)
    df.loc[:, "demand"] = ensys_opt.sequences_demand
    df.loc[:, "renewable"] = ensys_opt.sequences_inverter
    df.loc[:, "non_renewable"] = ensys_opt.sequences_genset
    df.loc[:, "surplus"] = ensys_opt.sequences_surplus
    df.to_csv(full_path_demand_coverage, index=False, float_format="%.3f")

    # store duration curves
    df = pd.read_csv(full_path_duration_curves)
    df.loc[:, "diesel_genset_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_genset) + 1)
            / len(ensys_opt.sequences_genset)
    )
    df.loc[:, "diesel_genset_duration"] = (
            100
            * np.sort(ensys_opt.sequences_genset)[::-1]
            / ensys_opt.sequences_genset.max()
    )
    df.loc[:, "pv_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_pv) + 1)
            / len(ensys_opt.sequences_pv)
    )
    df.loc[:, "pv_duration"] = (
            100 * np.sort(ensys_opt.sequences_pv)[::-1] / ensys_opt.sequences_pv.max()
    )
    df.loc[:, "rectifier_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_rectifier) + 1)
            / len(ensys_opt.sequences_rectifier)
    )
    df.loc[:, "rectifier_duration"] = 100 * np.nan_to_num(
        np.sort(ensys_opt.sequences_rectifier)[::-1]
        / ensys_opt.sequences_rectifier.max()
    )
    df.loc[:, "inverter_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_inverter) + 1)
            / len(ensys_opt.sequences_inverter)
    )
    df.loc[:, "inverter_duration"] = (
            100
            * np.sort(ensys_opt.sequences_inverter)[::-1]
            / ensys_opt.sequences_inverter.max()
    )
    df.loc[:, "battery_charge_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_battery_charge) + 1)
            / len(ensys_opt.sequences_battery_charge)
    )
    df.loc[:, "battery_charge_duration"] = (
            100
            * np.sort(ensys_opt.sequences_battery_charge)[::-1]
            / ensys_opt.sequences_battery_charge.max()
    )
    df.loc[:, "battery_discharge_percentage"] = (
            100
            * np.arange(1, len(ensys_opt.sequences_battery_discharge) + 1)
            / len(ensys_opt.sequences_battery_discharge)
    )
    df.loc[:, "battery_discharge_duration"] = (
            100
            * np.sort(ensys_opt.sequences_battery_discharge)[::-1]
            / ensys_opt.sequences_battery_discharge.max()
    )
    df.to_csv(full_path_duration_curves, index=False, float_format="%.3f")


@app.post("/shs_identification/")
def identify_shs(shs_identification_request: models.ShsIdentificationRequest):
    print("starting shs_identification...")

    # res = db.execute("select * from nodes")
    # nodes = res.fetchall()

    if len(nodes) == 0:
        return {
            "code": "success",
            "message": "No nodes in table, no identification to be performed",
        }

    # use latitude of the node that is the most west to set origin of x coordinates
    ref_latitude = min([node[1] for node in nodes])
    # use latitude of the node that is the most south to set origin of y coordinates
    ref_longitude = min([node[2] for node in nodes])

    nodes_df = shs_ident.create_nodes_df()

    cable_price_per_meter = (
        shs_identification_request.cable_price_per_meter_for_shs_mst_identification
    )
    additional_price_for_connection_per_node = (
        shs_identification_request.connection_cost_to_minigrid
    )

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


# -------------------------- FUNCTION FOR DEBUGGING-------------------------- #


def debugging_mode():
    """
    if host="0.0.0.0" and port=8000 does not work, the following can be used:
        host="127.0.0.1", port=8080
    """
    uvicorn.run(app, host="0.0.0.0", port=8080)
