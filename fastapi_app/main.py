import uuid
import ast
import asyncio
import math
import json
import base64
import random
from captcha.image import ImageCaptcha
import pandas as pd
import numpy as np
from io import StringIO
import fastapi_app.io.schema
from celery import group
from celery_worker import worker
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Any, Dict, List, Union
import time
import fastapi_app.tools.boundary_identification as bi
import fastapi_app.tools.coordinates_conversion as conv
import fastapi_app.tools.shs_identification as shs_ident
import fastapi_app.io.db.models as models
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from fastapi_app.tools.grids import Grid
from fastapi_app.tools.optimizer import Optimizer, GridOptimizer, EnergySystemOptimizer, po
from fastapi_app.tools.account_helpers import Hasher, create_guid, is_valid_credentials, send_activation_link, activate_mail, \
    authenticate_user, create_access_token, send_mail
from fastapi_app.tools import account_helpers as accounts
from fastapi_app.io.db import config, inserts, queries, sync_queries, sync_inserts
from fastapi_app.io.df_to_excel import df_to_xlsx
import pyutilib.subprocess.GlobalData
from fastapi_app.tools.solar_potential import get_dc_feed_in_sync_db_query
from fastapi_app.tools.error_logger import logger as error_logger

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False



app = FastAPI()

app.mount("/fastapi_app/static", StaticFiles(directory="fastapi_app/static"), name="static")
templates = Jinja2Templates(directory="fastapi_app/pages")

# define the template for importing json data in the form of arrays from js to python
json_object = Dict[Any, Any]
json_array = List[Any]
import_structure = Union[json_array, json_object]

"""
@app.on_event("startup")
def startup_event():
    if not sync_queries.check_if_weather_data_exists():
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data1.nc')
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data2.nc')
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data3.nc')
"""

@app.get('/insert_weather_data')
async def insert_weather_data(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user.is_superuser:
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data1.nc')
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data2.nc')
        sync_inserts.dump_weather_data_into_db('ERA5_weather_data3.nc')
    else:
        print('no privileges to insert weather data')


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    try:
        user = await accounts.get_user_from_cookie(request)
        user_name = user.email
    except:
        user_name = 'unknown username'
    error_logger.error_log(exc, request, user_name)
    return RedirectResponse(url="/?internal_error", status_code=303)

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    path = "fastapi_app/static/assets/favicon/favicon.ico"
    return FileResponse(path)


captcha_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.get('/get_captcha')
async def captcha(request: Request):
    captcha_text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
    captcha = ImageCaptcha()
    loop = asyncio.get_running_loop()
    captcha_data = await loop.run_in_executor(None, captcha.generate, captcha_text)
    base64_image = base64.b64encode(captcha_data.getvalue()).decode('utf-8')
    hashed_captcha = captcha_context.hash(captcha_text)
    return JSONResponse({'img': base64_image, 'hashed_captcha': hashed_captcha})


@app.get("/test_run")
async def test_run(request: Request):
    task = task_grid_opt.delay(9, 1)
    supply = False
    for i in range(1000):
        await asyncio.sleep(20)
        status = get_status_of_task(task.id)
        if status in ['success', 'failure', 'revoked']:
            if supply is True:
                break
            task = task_supply_opt.delay(9, 1)
            supply = True


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = await accounts.get_user_from_cookie(request)
    consent = request.cookies.get("consent_cookie")
    if user is None or consent is None:
        return templates.TemplateResponse("landing-page.html",
                                          {"request": request,
                                           'MAX_DAYS': int(os.environ.get('MAX_DAYS', 14)),
                                           'MAX_CONSUMER_ANONYMOUS': int(
                                               os.environ.get('MAX_CONSUMER_ANONYMOUS', 150))})
    else:
        projects = await queries.get_project_of_user(user.id)
        for project in projects:
            project.created_at = project.created_at.date()
            project.updated_at = project.updated_at.date()
            if user.task_id is not None and project.project_id == user.project_id:
                if bool(os.environ.get('DOCKERIZED')):
                    status = worker.AsyncResult(user.task_id).status.lower()
                else:
                    status = 'success'
                if status in ['success', 'failure', 'revoked']:
                    project_setup = await queries.get_model_instance(models.ProjectSetup, user.id, user.project_id)
                    user.task_id = ''
                    user.project_id = None
                    await inserts.update_model_by_user_id(user)
                    if status == 'success':
                        project_setup.status = "finished"
                    else:
                        project_setup.status = status
                    await inserts.merge_model(project_setup)
                    user.task_id = ''
                    user.project_id = None
                    await inserts.update_model_by_user_id(user)
        return templates.TemplateResponse("user_projects.html", {"request": request,
                                                                 'projects': projects})


@app.get("/project_setup", response_class=HTMLResponse)
async def project_setup(request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_id = request.query_params.get('project_id')
    if project_id is None:
        project_id = await queries.next_project_id_of_user(user.id)
    max_days = int(os.environ.get('MAX_DAYS', 365))
    return templates.TemplateResponse("project-setup.html", {"request": request,
                                                             'project_id': project_id,
                                                             'max_days': max_days})


@app.get("/user_registration", response_class=HTMLResponse)
async def user_registration(request: Request):
    return templates.TemplateResponse("user-registration.html", {"request": request})


@app.get("/imprint", response_class=HTMLResponse)
async def imprint(lang: str, request: Request):
    if lang == "en":
        lang = 'en_US'
    else:
        lang = 'de_DE'
    return templates.TemplateResponse("legal_notes.html", {"request": request,
                                                           "language": lang,
                                                           "page": 'imprint'})


@app.get("/privacy", response_class=HTMLResponse)
async def privacy(lang: str, request: Request):
    if lang == "en":
        lang = 'en_US'
    else:
        lang = 'de_DE'
    return templates.TemplateResponse("legal_notes.html", {"request": request,
                                                           "language": lang,
                                                           "page": 'privacy'})


@app.get("/activation_mail")
async def activation_mail(guid: str, request: Request):
    if not isinstance(guid, str):
        guid = request.path_params.get('guid')
    if guid is not None:
        await activate_mail(guid)
    return RedirectResponse(config.DOMAIN)


@app.get("/reset_password", response_class=HTMLResponse)
async def reset_password(guid, request: Request):
    if guid is not None:
        user = await queries.get_user_by_guid(guid)
        if user is not None:
            return templates.TemplateResponse("reset_password.html", {"request": request, 'guid': guid})
    return RedirectResponse('/')


@app.post("/reset_password")
async def reset_password(request: Request, form_data: Dict[str, str]):
    guid = form_data.get('guid')
    password = form_data.get('password')
    if guid is not None:
        user = await queries.get_user_by_guid(guid)
        if user is not None:
            if accounts.is_valid_password(password):
                user.hashed_password = Hasher.get_password_hash(password)
                user.guid = ''
                await inserts.merge_model(user)
                res = 'Password changed successfully.'
                validation = True
            else:
                validation = False
                res = 'The password needs to be at least 8 characters long'
            return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.get("/account_overview")
async def account_overview(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user is None or 'anonymous__' in user.email:
        return RedirectResponse('/')
    else:
        return templates.TemplateResponse("account_overview.html", {"request": request,
                                                                    'email': user.email})


@app.get("/consumer_selection")
async def consumer_selection(request: Request):
    project_id = request.query_params.get('project_id')
    try:
        int(project_id)
    except (TypeError, ValueError):
        RedirectResponse('/')
    return templates.TemplateResponse("consumer-selection.html", {"request": request, 'project_id': project_id})


@app.get("/consumer_types")
async def consumer_types(request: Request):
    project_id = request.query_params.get('project_id')
    try:
        int(project_id)
    except (TypeError, ValueError):
        RedirectResponse('/')
    return templates.TemplateResponse("consumer-types.html", {"request": request, 'project_id': project_id})


@app.get("/grid_design", response_class=HTMLResponse)
async def grid_design(request: Request):
    project_id = request.query_params.get('project_id')
    return templates.TemplateResponse("grid-design.html", {"request": request, 'project_id': project_id})


@app.post("/remove_project/{project_id}")
async def remove_project(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    if hasattr(user, 'id'):
        await inserts.remove_project(user.id, project_id)


@app.get("/demand_estimation", response_class=HTMLResponse)
async def demand_estimation(request: Request):
    project_id = request.query_params.get('project_id')
    return templates.TemplateResponse("demand_estimation.html", {"request": request, 'project_id': project_id})


@app.get("/energy_system_design", response_class=HTMLResponse)
async def energy_system_design(request: Request):
    project_id = request.query_params.get('project_id')
    return templates.TemplateResponse("energy-system-design.html", {"request": request, 'project_id': project_id})


@app.get("/simulation_results", response_class=HTMLResponse)
async def simulation_results(request: Request):
    project_id = request.query_params.get('project_id')
    try:
        int(project_id)
    except (TypeError, ValueError) as e:
        raise e
    return templates.TemplateResponse("simulation-results.html", {"request": request, 'project_id': project_id})


@app.get("/calculating")
async def calculating(request: Request):
    project_id = request.query_params.get('project_id')
    user = await accounts.get_user_from_cookie(request)
    try:
        int(project_id)
    except (TypeError, ValueError):
        return RedirectResponse('/')
    if 'anonymous' in user.email:
        msg = 'You will be forwarded after the model calculation is completed.'
        email_opt = False
    elif user.task_id is not None and len(user.task_id) > 20 and not task_is_finished(user.task_id):
        msg = 'CAUTION: You have a calculation in progress that has not yet been completed. Therefore you cannot' \
              ' start another calculation. You can cancel the already running calculation by clicking on the' \
              ' following button:'
        return templates.TemplateResponse("calculating.html", {"request": request,
                                                               'project_id': project_id,
                                                               'msg': msg,
                                                               'task_id': user.task_id,
                                                               'time': 0,
                                                               'email_opt': True})
    else:
        msg = 'You will be forwarded after the model calculation is completed. You can also close the window and view' \
              ' the results in your user account after the calculation is finished.'
        email_opt = True
    return templates.TemplateResponse("calculating.html", {"request": request,
                                                           'project_id': project_id,
                                                           'msg': msg,
                                                           'email_opt': email_opt})


@app.post("/set_email_notification/{project_id}/{is_active}")
async def set_email_notification(project_id: int, is_active: bool, request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_setup = await queries.get_model_instance(models.ProjectSetup, user.id, project_id)
    project_setup.email_notification = is_active
    await inserts.merge_model(project_setup)


@app.get("/get_demand_coverage_data/{project_id}")
async def get_demand_coverage_data(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.DemandCoverage, user.id, project_id, is_timeseries=True)
    df = df.reset_index(drop=True)
    return json.loads(df.to_json())


@app.get("/db_links_to_js/{project_id}")
async def db_links_to_js(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    links_json = await queries.get_links_json(user.id, project_id)
    return links_json


@app.get("/db_nodes_to_js/{project_id}/{markers_only}")
async def db_nodes_to_js(project_id: str, markers_only: bool, request: Request):
    user = await accounts.get_user_from_cookie(request)
    if project_id == 'undefined':
        project_id = get_project_id_from_request(request)
    df = await queries.get_df(models.Nodes, user.id, project_id)
    if not df.empty:
        df = df[['latitude', 'longitude', 'how_added', 'node_type', 'surface_area', 'consumer_type', 'consumer_detail']]
        if markers_only is True:
            df = df[df['node_type'] == 'consumer']
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        nodes_list = df.to_dict('records')
        return nodes_list
    else:
        return None


@app.post("/consumer_to_db/{project_id}")
async def consumer_to_db(project_id: str, map_elements: fastapi_app.io.schema.MapDataRequest, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = pd.DataFrame.from_records(map_elements.map_elements)
    if df.empty is True:
        await inserts.remove(models.Nodes, user.id, project_id)
        return
    df = df[df['node_type'] == 'consumer']
    if df.empty is True:
        await inserts.remove(models.Nodes, user.id, project_id)
        return
    df = df[['latitude', 'longitude', 'how_added', 'node_type', 'surface_area', 'consumer_type', 'consumer_detail']]
    df.loc[df['surface_area'] == 0, 'surface_area'] = 10 + np.random.uniform(0, 1)
    df['surface_area'] = df['surface_area'].fillna(0)
    df['consumer_type'] = df['consumer_type'].fillna('household')
    df['consumer_detail'] = df['consumer_detail'].fillna('default')
    df['is_connected'] = True
    df['peak_demand'] = 0
    df['average_consumption'] = 0
    known_surface_df = df[df['surface_area'] > 0]
    unknown_surface_df = df[df['surface_area'] == 0]
    known_surface_df = demand_estimation2(known_surface_df)
    df = pd.concat([known_surface_df, unknown_surface_df])
    df = df.round(decimals=6)
    if df.empty:
        await inserts.remove(models.Nodes, user.id, project_id)
        return
    df["node_type"] = df["node_type"].astype(str)
    df['surface_area'] = df.surface_area.map(lambda x: "%.2f" % x)
    df['peak_demand'] = df.peak_demand.map(lambda x: "%.3f" % x)
    df['average_consumption'] = df.average_consumption.map(lambda x: "%.3f" % x)
    if len(df.index) != 0:
        if 'parent' in df.columns:
            df['parent'] = df['parent'].replace('unknown', None)
    await inserts.insert_nodes_df(df, user.id, project_id)


@app.get("/load_results/{project_id}")
async def load_results(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.Results, user.id, project_id)
    if df.empty:
        return {}
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
                 'max_voltage_drop': '%',
                 'shortage_total': '%',
                 'surplus_rate': '%',
                 'time': 's',
                 'co2_savings': 't/a'}
    df = df[list(unit_dict.keys())].round(1).astype(str)
    for col in df.columns:
        if unit_dict[col] in ['%', 's']:
            df[col] = df[col].where(df[col] != 'None', 0)
            if df[col].isna().sum() == 0:
                df[col] = df[col].astype(float).round(1).astype(str)
        df[col] = df[col] + ' ' + unit_dict[col]
    results = df.to_dict(orient='records')[0]
    return results


@app.get("/load_previous_data/{page_name}")
async def load_previous_data(page_name, request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_id = request.query_params.get('project_id')
    if page_name == "project_setup":
        if project_id == 'new':
            project_id = await queries.next_project_id_of_user(user.id)
            return models.ProjectSetup(project_id=project_id)
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        project_setup = await queries.get_model_instance(models.ProjectSetup, user.id, project_id)
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
        grid_design = await queries.get_model_instance(models.GridDesign, user.id, project_id)
        return grid_design
    elif page_name == "demand_estimation":
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        demand_estimation = await queries.get_model_instance(models.Demand, user.id, project_id)
        demand_estimation.maximum_peak_load = str(demand_estimation.maximum_peak_load) \
            if demand_estimation.maximum_peak_load is not None else ''
        demand_estimation.average_daily_energy = str(demand_estimation.average_daily_energy) \
            if demand_estimation.average_daily_energy is not None else ''
        demand_estimation.custom_calibration = True \
            if len(demand_estimation.maximum_peak_load) > 0 or len(demand_estimation.average_daily_energy) \
            else False
        return demand_estimation


@app.post("/add_user_to_db/")
async def add_user_to_db(data: Dict[str, str]):
    email = data.get('email')
    password = data.get('password')

    class User:
        def __init__(self, email: str, password: str):
            self.email = email
            self.password = password

    user = User(email=email, password=password)
    captcha_input = data.get('captcha_input')
    hashed_captcha = data.get('hashed_captcha')
    res = await is_valid_credentials(user)
    if res[0] is True:
        if captcha_context.verify(captcha_input, hashed_captcha):
            guid = create_guid()
            user = models.User(email=email,
                               hashed_password=Hasher.get_password_hash(password),
                               guid=guid,
                               is_confirmed=False,
                               is_active=False,
                               is_superuser=False)
            await inserts.merge_model(user)
            send_activation_link(user.email, guid)
        else:
            res = [False, 'Please enter a valid captcha']
    return fastapi_app.io.schema.ValidRegistration(validation=res[0], msg=res[1])


@app.post("/anonymous_login/")
async def anonymous_login(data: Dict[str, str], response: Response):
    captcha_input = data.get('captcha_input')
    hashed_captcha = data.get('hashed_captcha')
    if captcha_context.verify(captcha_input, hashed_captcha):
        guid = str(uuid.uuid4())
        name = 'anonymous__' + guid
        user = models.User(email=name,
                           hashed_password='',
                           guid='',
                           is_confirmed=True,
                           is_active=True,
                           is_superuser=False,
                           task_id='',
                           project_id=0)
        await inserts.merge_model(user)
        user = await queries.get_user_by_username(name)
        access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS)
        access_token = create_access_token(data={"sub": name}, expires_delta=access_token_expires)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}",
                            httponly=True)  # set HttpOnly cookie in response
        if bool(os.environ.get('DOCKERIZED')):
            minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS + 60
            eta = datetime.utcnow() + timedelta(minutes=minutes)
            task_remove_anonymous_users.apply_async((user.email, user.id,), eta=eta)
        validation, res = True, ''
    else:
        validation, res = False, 'Please enter a valid captcha'
    return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.post("/login/")
async def login(response: Response, credentials: fastapi_app.io.schema.Credentials):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        is_valid, res = await authenticate_user(credentials.email, credentials.password)
        if is_valid:
            if credentials.remember_me:
                access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED)
            else:
                access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
            del credentials
            access_token = create_access_token(data={"sub": res.email}, expires_delta=access_token_expires)
            response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
            return fastapi_app.io.schema.ValidRegistration(validation=True, msg="")
        else:
            del credentials
            return fastapi_app.io.schema.ValidRegistration(validation=False, msg=res)


@app.post("/consent_cookie/")
async def consent_cookie(response: Response):
    expire_date = datetime.utcnow()
    expire_date = expire_date + timedelta(days=365)
    expire_date = expire_date.replace(tzinfo=timezone.utc)
    response.set_cookie(key="consent_cookie", value='true', httponly=True, expires=expire_date)


@app.post("/change_email/")
async def change_email(request: Request, credentials: fastapi_app.io.schema.Credentials):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        user = await accounts.get_user_from_cookie(request)
        is_valid, res = await authenticate_user(user.email, credentials.password)
        validation = False
        if is_valid:
            user.email = credentials.email
            if accounts.is_valid_email(user):
                user.is_confirmed = False
                user.guid = create_guid()
                await inserts.merge_model(user)
                send_activation_link(credentials.email, user.guid)
                res = 'Please click the activation link we sent to your email.'
                validation = True
            else:
                res = 'Please enter a valid email address.'
        else:
            del credentials
        return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.post("/change_pw/")
async def change_pw(request: Request, passwords: fastapi_app.io.schema.ChangePW):
    user = await accounts.get_user_from_cookie(request)
    is_valid, res = await authenticate_user(user.email, passwords.old_password)
    validation = False
    if is_valid:
        if accounts.is_valid_password(passwords.new_password):
            user.hashed_password = Hasher.get_password_hash(passwords.new_password)
            await inserts.merge_model(user)
            res = 'Password changed successfully.'
            validation = True
        else:
            res = 'The password needs to be at least 8 characters long'
    else:
        del passwords
    return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.post("/send_reset_password_email/")
async def send_reset_password_email(data: Dict[str, str], request: Request):
    email = data.get('email')
    captcha_input = data.get('captcha_input')
    hashed_captcha = data.get('hashed_captcha')
    user = await queries.get_user_by_username(email)
    if user is None:
        validation, res = False, 'Email address is not registered'
    else:
        if captcha_context.verify(captcha_input, hashed_captcha):
            guid = str(uuid.uuid4()).replace('-', '')[:24]
            user.guid = guid
            await inserts.merge_model(user)
            msg = 'For your PeopleSuN-Account a password reset was requested. If you did not request a password reset, ' \
                  'please ignore this email. Otherwise, please click the following link:\n\n{}/reset_password?guid={}' \
                .format(config.DOMAIN, guid)
            send_mail(email, msg, 'PeopleSuN-Account: Reset your password')
            validation, res = True, 'Please click the link we sent to your email.'
        else:
            validation, res = False, 'Please enter a valid captcha'
    return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.post("/delete_account/")
async def change_pw(response: Response, request: Request, form_data: fastapi_app.io.schema.Password):
    user = await accounts.get_user_from_cookie(request)
    is_valid, res = await authenticate_user(user.email, form_data.password)
    validation = False
    if is_valid:
        await inserts.remove_account(user.email, user.id)
        response.delete_cookie("access_token")
        res = 'Account removed'
        validation = True
    return fastapi_app.io.schema.ValidRegistration(validation=validation, msg=res)


@app.post("/logout/")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "success"}


@app.post("/query_account_data/")
async def query_account_data(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user is not None:
        name = user.email
        if 'anonymous__' in name:
            name = name.split('__')[0]
        return fastapi_app.io.schema.UserOverview(email=name)
    else:
        return fastapi_app.io.schema.UserOverview(email="")


@app.post("/has_cookie/")
async def has_cookie(request: Request, has_cookies: fastapi_app.io.schema.HasCookies):
    if has_cookies.access_token:
        token = request.cookies.get("access_token")
        if token is None:
            return False
    if has_cookies.consent_cookie:
        consent = request.cookies.get("consent_cookie")
        if consent is None:
            return False
    return True


@app.post("/save_grid_design/")
async def save_grid_design(request: Request, data: fastapi_app.io.schema.SaveGridDesign):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    data.grid_design['id'] = user.id
    data.grid_design['project_id'] = project_id
    grid_design = models.GridDesign(**data.grid_design)
    grid_design.allow_shs = ast.literal_eval(grid_design.allow_shs)
    await inserts.merge_model(grid_design)


@app.post("/save_demand_estimation/")
async def save_demand_estimation(request: Request, data: fastapi_app.io.schema.SaveDemandEstimation):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    custom_calibration = ast.literal_eval(data.demand_estimation['custom_calibration'])
    if custom_calibration is None or '':
        maximum_peak_load = None
        average_daily_energy = None
    else:
        try:
            maximum_peak_load = round(float(data.demand_estimation['maximum_peak_load']), 1)
        except ValueError:
            maximum_peak_load = None
        try:
            average_daily_energy = round(float(data.demand_estimation['average_daily_energy']), 1)
        except ValueError:
            average_daily_energy = None
    dictionary = {'id': user.id,
                  'project_id': project_id,
                  'household_option': data.demand_estimation['household_option'],
                  'maximum_peak_load': maximum_peak_load,
                  'average_daily_energy': average_daily_energy}
    demand_estimation = models.Demand(**dictionary)
    await inserts.merge_model(demand_estimation)


@app.post("/save_project_setup/{project_id}")
async def save_project_setup(project_id, request: Request, data: fastapi_app.io.schema.SaveProjectSetup):
    user = await accounts.get_user_from_cookie(request)
    # project_id = get_project_id_from_request(request)
    timestamp = pd.Timestamp.now()
    data.page_setup['created_at'] = timestamp
    data.page_setup['updated_at'] = timestamp
    data.page_setup['id'] = user.id
    data.page_setup['project_id'] = project_id
    project_setup = models.ProjectSetup(**data.page_setup)
    await inserts.merge_model(project_setup)


@app.post("/save_energy_system_design/")
async def save_energy_system_design(request: Request, data: fastapi_app.io.schema.OptimizeEnergySystemRequest):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    df = data.to_df()
    await inserts.insert_energysystemdesign_df(df, user.id, project_id)


def get_project_id_from_request(request: Request):
    project_id = request.query_params.get('project_id')
    if project_id is None:
        try:
            project_id = [tup[1].decode() for tup in request.scope['headers']
                          if 'project_id' in tup[1].decode()][0].split('=')[-1]
        except IndexError:
            print(request.scope['headers'])
    project_id = int(project_id)
    return project_id


@app.get("/get_optimal_capacities/{project_id}")
async def get_optimal_capacities(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.Results, user.id, project_id)
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
async def get_lcoe_breakdown(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.Results, user.id, project_id)
    lcoe_breakdown = {}
    lcoe_breakdown["renewable_assets"] = str(df.loc[0, "cost_renewable_assets"])
    lcoe_breakdown["non_renewable_assets"] = str(df.loc[0, "cost_non_renewable_assets"])
    lcoe_breakdown["grid"] = str(df.loc[0, "cost_grid"])
    lcoe_breakdown["fuel"] = str(df.loc[0, "cost_fuel"])
    # importing nodes and links from the csv files to the map
    return lcoe_breakdown


@app.get("/get_data_for_sankey_diagram/{project_id}")
async def get_data_for_sankey_diagram(project_id, request: Request):
    sankey_data = {}
    user = await  accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.Results, user.id, project_id)
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
async def get_data_for_energy_flows(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.EnergyFlow, user.id, project_id)
    df = df.reset_index(drop=True)
    return json.loads(df.to_json())


@app.get("/get_data_for_duration_curves/{project_id}")
async def get_data_for_duration_curves(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.DurationCurve, user.id, project_id)
    return json.loads(df.to_json())


@app.get("/get_co2_emissions_data/{project_id}")
async def get_co2_emissions_data(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await queries.get_df(models.Emissions, user.id, project_id)
    return json.loads(df.to_json())


@app.post("/add_buildings_inside_boundary")
async def add_buildings_inside_boundary(js_data: fastapi_app.io.schema.MapData, request: Request):
    user = await accounts.get_user_from_cookie(request)
    boundary_coordinates = js_data.boundary_coordinates[0][0]
    df = pd.DataFrame.from_dict(boundary_coordinates).rename(columns={'lat': 'latitude', 'lng': 'longitude'})
    if df['latitude'].max() - df['latitude'].min() > float(os.environ.get("MAX_LAT_LON_DIST", 0.15)):
        return JSONResponse({'executed': False,
                             'msg': 'The maximum latitude distance selected is too large. '
                                    'Please select a smaller area.'})
    elif df['longitude'].max() - df['longitude'].min() > float(os.environ.get("MAX_LAT_LON_DIST", 0.15)):
        return JSONResponse({'executed': False,
                             'msg': 'The maximum longitude distance selected is too large. '
                                    'Please select a smaller area.'})
    data, building_coordidates_within_boundaries, building_area\
        = bi.get_consumer_within_boundaries(df)
    if building_coordidates_within_boundaries is None:
        return JSONResponse({'executed': False, 'msg': 'In the selected area, no buildings could be identified.'})
    nodes = defaultdict(list)
    for label, coordinates in building_coordidates_within_boundaries.items():
        nodes["latitude"].append(coordinates[0])
        nodes["longitude"].append(coordinates[1])
        nodes["how_added"].append("automatic")
        nodes["node_type"].append("consumer")
        nodes["consumer_type"].append('household')
        nodes["consumer_detail"].append('default')
        nodes["surface_area"].append(building_area[label])
    if user.email.split('__')[0] == 'anonymous':
        max_consumer = int(os.environ.get("MAX_CONSUMER_ANONNYMOUS", 150))
    else:
        max_consumer = int(os.environ.get("MAX_CONSUMER", 1000))
    if len(nodes['latitude']) > max_consumer:
        return JSONResponse({'executed': False,
                             'msg': 'You have selected {} consumers. You can select a maximum of {} consumer. '
                                    'Reduce the number of consumers by selecting a small area, for example.'
                            .format(len(data['elements']), max_consumer)})
    df = pd.DataFrame.from_dict(nodes)
    df_exisiting = pd.DataFrame.from_records(js_data.map_elements)
    df = pd.concat([df, df_exisiting], ignore_index=True)
    df['surface_area'] = df['surface_area'].round(2)
    df = df.drop_duplicates()
    nodes_list = df.to_dict('records')
    return JSONResponse({'executed': True, 'msg': '', 'new_consumers': nodes_list})


@app.post("/remove_buildings_inside_boundary")
async def remove_buildings_inside_boundary(data: fastapi_app.io.schema.MapData):
    df = pd.DataFrame.from_records(data.map_elements)
    if not df.empty:
        boundaries = pd.DataFrame.from_records(data.boundary_coordinates[0][0]).values.tolist()
        df['inside_boundary'] = bi.are_points_in_boundaries(df, boundaries=boundaries, )
        df = df[df['inside_boundary'] == False]
        df = df.drop(columns=['inside_boundary'])
        return JSONResponse({'map_elements': df.to_dict('records')})


# add new manually-selected nodes to the *.csv file
@app.post("/database_add_remove_manual/{add_remove}/{project_id}")
async def database_add_remove_manual(add_remove: str, project_id, add_node_request: fastapi_app.io.schema.AddNodeRequest,
                                     request: Request):
    user = await accounts.get_user_from_cookie(request)
    nodes = models.Nodes(**dict(add_node_request)).to_dict()
    if add_remove == "remove":
        df = await queries.get_df(models.Nodes, user.id, project_id)
        df = df[(df["node_type"] != "pole") & (df["node_type"] != "power-house")]
        for index in df.index:
            if (round(add_node_request.latitude, 6) == df.to_dict()["latitude"][index]) and \
                    (round(add_node_request.longitude, 6) == df.to_dict()["longitude"][index]):
                df.drop(labels=index, axis=0, inplace=True)
        df = df.reset_index(drop=True)
        await inserts.update_nodes_and_links(True, False, df.to_dict(), user.id, project_id, add=False)
    else:
        await inserts.update_nodes_and_links(True, False, nodes, user.id, project_id, add=True, replace=False)


def _demand_estimation(nodes, update_total_demand):
    # after collecting all surface areas, based on a simple assumption, the peak demand will be obtained
    max_surface_area = max(nodes["surface_area"])

    # normalized demands is a CSV file with 5 columns representing the very low to very high demand profiles
    normalized_demands = pd.read_csv(config.full_path_demands, delimiter=";", header=None)

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
        timeseries = pd.read_csv(config.full_path_timeseries)
        # replace the demand column in the timeseries file with the total demand calculated here
        timeseries["Demand"] = total_demand
        # update the CSV file
        timeseries.to_csv(config.full_path_timeseries, index=False)
    else:
        for area in nodes["surface_area"]:
            if area <= 0.2 * max_surface_area:
                nodes["peak_demand"] = 0.01 * area
            elif area < 0.4 * max_surface_area:
                nodes["peak_demand"] = 0.02 * area
            elif area < 0.6 * max_surface_area:
                nodes["peak_demand"] = 0.03 * area
            elif area < 0.8 * max_surface_area:
                nodes["peak_demand"] = 0.04 * area
            else:
                nodes["peak_demand"] = 0.05 * area
        max_peak_demand = max(nodes["peak_demand"])
        counter = 0
        for peak_demand in nodes["peak_demand"]:
            if peak_demand <= 0.2 * max_peak_demand:
                nodes["average_consumption"] = normalized_demands.iloc[:, 0].sum() * nodes["peak_demand"][counter]
            elif peak_demand < 0.4 * max_peak_demand:
                nodes["average_consumption"] = normalized_demands.iloc[:, 1].sum() * nodes["peak_demand"][counter]
            elif peak_demand < 0.6 * max_peak_demand:
                nodes["average_consumption"] = normalized_demands.iloc[:, 2].sum() * nodes["peak_demand"][counter]
            elif peak_demand < 0.8 * max_peak_demand:
                nodes["average_consumption"] = normalized_demands.iloc[:, 3].sum() * nodes["peak_demand"][counter]
            else:
                nodes["average_consumption"] = normalized_demands.iloc[:, 4].sum() * nodes["peak_demand"][counter]
            counter += 1
        return nodes

def demand_estimation2(df):
    max_surface_area = df["surface_area"].max()

    for factor in [0.2, 0.4, 0.6, 0.8]:
        df.loc[df['surface_area'].between((factor - 0.2) * max_surface_area,
                                          factor * max_surface_area), 'peak_demand'] = 0.05 * df['surface_area']
    df.loc[df['surface_area'] >= factor * 0.8 * max_surface_area, 'peak_demand'] = factor / 20 * df['surface_area']
    normalized_demands = pd.read_csv(config.full_path_demands, delimiter=";", header=None)
    max_peak_demand = df["peak_demand"].max()
    for i, factor in enumerate([0.2, 0.4, 0.6, 0.8]):
        df.loc[df['peak_demand'].between((factor - 0.2) * max_peak_demand,
                                          factor * max_surface_area), 'average_consumption'] \
            = normalized_demands.iloc[:, i].sum() * df["peak_demand"]

    df.loc[df['peak_demand'] >= 0.8 * max_peak_demand, 'average_consumption'] \
        = normalized_demands.iloc[:, 4].sum() * df["peak_demand"]
    return df


async def remove_results(user_id, project_id):
    await inserts.remove(models.Results, user_id, project_id)
    await inserts.remove(models.DemandCoverage, user_id, project_id)
    await inserts.remove(models.EnergyFlow, user_id, project_id)
    await inserts.remove(models.Emissions, user_id, project_id)
    await inserts.remove(models.DurationCurve, user_id, project_id)
    await inserts.remove(models.Links, user_id, project_id)


@worker.task(name='celery_worker.task_grid_opt', force=True, track_started=True)
def task_grid_opt(user_id, project_id):
    result = optimize_grid(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_supply_opt', force=True, track_started=True)
def task_supply_opt(user_id, project_id):
    result = optimize_energy_system(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_remove_anonymous_users', force=True, track_started=True)
def task_remove_anonymous_users(user_email, user_id):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        result = loop.run_until_complete(inserts.remove_account(user_email, user_id))
    else:
        result = asyncio.run(inserts.remove_account(user_email, user_id))
    return result

async def optimization(user_id, project_id):
    await remove_results(user_id, project_id)
    project_setup = await queries.get_model_instance(models.ProjectSetup, user_id, project_id)
    project_setup.status = "queued"
    await inserts.merge_model(project_setup)
    if bool(os.environ.get('DOCKERIZED')):
        task = task_grid_opt.delay(user_id, project_id)
        return task.id
    else:
        optimize_grid(user_id, project_id)
        optimize_energy_system(user_id, project_id)
        return 'no_celery_id'


def get_status_of_task(task_id):
    status = worker.AsyncResult(task_id).status.lower()
    return status


def task_is_finished(task_id):
    status = get_status_of_task(task_id)
    if status in ['success', 'failure', 'revoked']:
        return True
    else:
        return False


@app.post("/forward_if_no_task_is_pending")
async def forward_if_no_task_is_pending(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user.task_id is not None and len(user.task_id) > 20 and not task_is_finished(user.task_id):
        res = {'forward': False, 'task_id': user.task_id}
    else:
        res = {'forward': True, 'task_id': ''}
    return JSONResponse(res)


@app.post("/forward_if_consumer_selection_exists/{project_id}")
async def forward_if_consumer_selection_exists(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    nodes = await queries.get_model_instance(models.Nodes, user.id, project_id)
    if nodes is not None and len(nodes.to_json()['consumer_type']) > 0:
        res = {'forward': True}
    else:
        res = {'forward': False}
    return JSONResponse(res)


@app.post("/start_calculation/{project_id}")
async def start_calculation(project_id, request: Request):
    if project_id is None:
        project_id = request.query_params.get('project_id')
    user = await accounts.get_user_from_cookie(request)
    task_id = await optimization(user.id, project_id)
    user.task_id = task_id
    user.project_id = int(project_id)
    await inserts.update_model_by_user_id(user)
    return JSONResponse({'task_id': task_id})


@app.post('/waiting_for_results/')
async def waiting_for_results(request: Request, data: fastapi_app.io.schema.TaskInfo):
    max_time = 3600 * 24 * 7
    t_wait = -2E-05 *  data.time + 0.0655 *  data.time + 5.7036 if data.time < 1800 else 60
    # ToDo: set the time limit based on number of queued tasks and size of the model
    res = {'time': int(data.time) + t_wait, 'status': '', 'task_id': data.task_id, 'model': data.model}
    if len(data.task_id) > 12 and max_time > res['time']:
        if not data.time == 0:
            await asyncio.sleep(t_wait)
        status = get_status_of_task(data.task_id)
        if status in ['success', 'failure', 'revoked']:
            res['finished'] = True
        else:
            res['finished'] = False
            if status in ['pending', 'received', 'retry']:
                res['status'] = "task queued, waiting for processing..."
            else:
                res['status'] = "spatial grid optimization is running..."
    else:
        res['finished'] = True
    if res['finished'] is True:
        user = await accounts.get_user_from_cookie(request)
        if data.model == 'grid' and bool(os.environ.get('DOCKERIZED')):
            task = task_supply_opt.delay(user.id, data.project_id)
            user.task_id = task.id
            await inserts.update_model_by_user_id(user)
            res['finished'] = False
            res['status'] = "power supply optimization is running..."
            res['model'] = 'supply'
            res[task.id] = task.id
        else:
            project_setup = await queries.get_model_instance(models.ProjectSetup, user.id, user.project_id)
            if project_setup is not None:
                if 'status' in locals():
                    if status == 'success':
                        project_setup.status = "finished"
                    else:
                        project_setup.status = status
                else:
                    project_setup.status = "finished"
                await inserts.merge_model(project_setup)
                user.task_id = ''
                user.project_id = None
                await inserts.update_model_by_user_id(user)
    return JSONResponse(res)


@app.post('/has_pending_task/{project_id}')
async def has_pending_task(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user.task_id is not None \
            and len(user.task_id) > 20 \
            and not task_is_finished(user.task_id) \
            and user.project_id == int(project_id):
        return JSONResponse({'has_pending_task': True})
    else:
        return JSONResponse({'has_pending_task': False})


@app.post('/revoke_task/')
async def revoke_task(request: Request, data: fastapi_app.io.schema.TaskID):
    celery_task = worker.AsyncResult(data.task_id)
    celery_task.revoke(terminate=True, signal='SIGKILL')
    user = await accounts.get_user_from_cookie(request)
    await remove_results(user.id, data.project_id)


@app.post('/revoke_users_task/')
async def revoke_users_task(request: Request):
    user = await accounts.get_user_from_cookie(request)
    celery_task = worker.AsyncResult(user.task_id)
    celery_task.revoke(terminate=True, signal='SIGKILL')
    user = await accounts.get_user_from_cookie(request)
    user.task_id = ''
    user.project_id = None
    await inserts.update_model_by_user_id(user)


def optimize_grid(user_id, project_id):
    try:
        # Grab Currrent Time Before Running the Code
        project_setup = sync_queries.get_project_setup_of_user(user_id, project_id)
        project_setup.status = "in progress"
        sync_inserts.merge_model(project_setup)
        start_execution_time = time.monotonic()
        # create GridOptimizer object
        df = sync_queries.get_input_df(user_id, project_id)
        opt = GridOptimizer(start_date=df.loc[0, "start_date"],
                            n_days=df.loc[0, "n_days"],
                            project_lifetime=df.loc[0, "project_lifetime"],
                            wacc=df.loc[0, "interest_rate"] / 100,
                            tax=0, )
        nodes = sync_queries.get_df(models.Nodes, user_id, project_id)

        if len(nodes) == 0:
            return {"code": "success", "message": "Empty grid cannot be optimized!"}

        # create a new "grid" object from the Grid class
        epc_distribution_cable = ((opt.crf *
                                   Optimizer.capex_multi_investment(
                                       opt,
                                       capex_0=df.loc[0, "distribution_cable_capex"],
                                       component_lifetime=df.loc[0, "distribution_cable_lifetime"], ))
                                  * opt.n_days / 365)

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
        demand_full_year = pd.read_csv(filepath_or_buffer=config.full_path_timeseries)
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
        _demand_estimation(nodes=grid.nodes, update_total_demand=True)

        n_poles = opt.find_opt_number_of_poles(grid, df.loc[0, "connection_cable_max_length"], n_mg_consumers)
        opt.determine_poles(grid=grid, min_n_clusters=n_poles)
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
        peak_demand_shs_consumers = grid.nodes[grid.nodes["is_connected"] == False].loc[:, "peak_demand"]
        cost_shs = epc_shs * peak_demand_shs_consumers.sum()

        # get all poles obtained by the network relaxation method
        poles = grid.poles().reset_index(drop=True)

        # remove the unnecessary columns to make it compatible with the CSV files
        # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
        poles.drop(labels=["x", "y", "cluster_label", "type_fixed", "n_connection_links", "n_distribution_links"],
                   axis=1,
                   inplace=True)


        # Store the list of poles in the "node" database.
        sync_inserts.update_nodes_and_links(True, False, poles.to_dict(), user_id, project_id)

        # get all links obtained by the network relaxation method
        links = grid.links.reset_index(drop=True)

        # remove the unnecessary columns to make it compatible with the CSV files
        # TODO: When some of these columns are removed in the future, this part here needs to be updated too.
        links.drop(labels=["x_from", "y_from", "x_to", "y_to", "n_consumers", "total_power", "from_node", "to_node"],
                   axis=1,
                   inplace=True)

        # store the list of poles in the "node" database
        sync_inserts.update_nodes_and_links(False, True, links.to_dict(), user_id, project_id)

        # Grab Currrent Time After Running the Code
        end_execution_time = time.monotonic()

        # store data for showing in the final results
        results = models.Results()
        results.n_consumers = len(grid.consumers())
        results.n_shs_consumers = n_shs_consumers
        results.n_poles = len(grid.poles())
        results.length_distribution_cable = int(grid.links[grid.links.link_type == "distribution"]["length"].sum())
        results.length_connection_cable = int(grid.links[grid.links.link_type == "connection"]["length"].sum())
        results.cost_grid = int(grid.cost())
        results.cost_shs = int(cost_shs)
        results.time_grid_design = round(end_execution_time - start_execution_time, 1)
        results.n_distribution_links = int(grid.links[grid.links["link_type"] == "distribution"].shape[0])
        results.n_connection_links = int(grid.links[grid.links["link_type"] == "connection"].shape[0])
        voltage_drop_df = grid.get_voltage_drop_at_nodes()
        voltage_drop = voltage_drop_df['voltage drop fraction [%]'].max()
        if isinstance(voltage_drop, float):
            results.max_voltage_drop = round(float(voltage_drop_df['voltage drop fraction [%]'].max()), 1)
        else:
            results.max_voltage_drop = 0
        df = results.to_df()

        sync_inserts.insert_results_df(df, user_id, project_id)
        #grid.allocate_consumers_and_poles_to_branches()
        #grid.sum_up_consumers_depending_on_each_pole()

        #grid.find_capacity_of_each_link()
        # ToDo:  what is with this last operations? It is not stored in the database.
        #grid.distribute_grid_cost_among_consumers()
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc


def optimize_energy_system(user_id, project_id):
    try:
        # Grab Currrent Time Before Running the Code
        start_execution_time = time.monotonic()
        df = sync_queries.get_input_df(user_id, project_id)
        energy_system_design = sync_queries.get_energy_system_design(user_id, project_id)
        solver = 'gurobi' if po.SolverFactory('gurobi').available() else 'cbc'
        nodes = sync_queries.get_df(models.Nodes, user_id, project_id)
        if not nodes[nodes['consumer_type'] == 'power_house'].empty:
            lat, lon = nodes[nodes['consumer_type'] == 'power_house']['latitude', 'longitude'].to_list()
        else:
            lat, lon = nodes[['latitude', 'longitude']].mean().to_list()
        n_days = min(df.loc[0, "n_days"], int(os.environ.get('MAX_DAYS', 365)))
        start = pd.to_datetime(df.loc[0, "start_date"])
        end = start + timedelta(days=int(n_days))
        solar_potential_df = get_dc_feed_in_sync_db_query(lat, lon, start, end)
        ensys_opt = EnergySystemOptimizer(
            start_date=df.loc[0, "start_date"],
            n_days=n_days,
            project_lifetime=df.loc[0, "project_lifetime"],
            wacc=df.loc[0, "interest_rate"] / 100,
            tax=0,
            solar_potential=solar_potential_df,
            path_data=config.full_path_timeseries,
            solver=solver,
            pv=energy_system_design['pv'],
            diesel_genset=energy_system_design['diesel_genset'],
            battery=energy_system_design['battery'],
            inverter=energy_system_design['inverter'],
            rectifier=energy_system_design['rectifier'],
            shortage=energy_system_design['shortage'], )
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
        df = df.round(3)
        sync_inserts.insert_df(models.Emissions, df, user_id, project_id)
        # TODO: -2 must actually be -1, but for some reason, the co2-emission csv file has an additional empty row
        co2_savings = df.loc[:, "co2_savings"][-2]  # takes the last element of the cumulative sum
        # store data for showing in the final results
        df = sync_queries.get_df(models.Results, user_id, project_id)
        df.loc[0, "cost_renewable_assets"] = ensys_opt.total_renewable
        df.loc[0, "cost_non_renewable_assets"] = ensys_opt.total_non_renewable
        df.loc[0, "cost_fuel"] = ensys_opt.total_fuel
        df.loc[0, "lcoe"] = (100 * (ensys_opt.total_revenue + df.loc[0, "cost_grid"]
                                    + df.loc[0, "cost_shs"]) / ensys_opt.total_demand)
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
        df.loc[0, "fuel_to_diesel_genset"] = (ensys_opt.sequences_fuel_consumption.sum() * 0.846 *
                                              ensys_opt.diesel_genset["parameters"]["fuel_lhv"] / 1000)
        df.loc[0, "diesel_genset_to_rectifier"] = (ensys_opt.sequences_rectifier.sum() /
                                                   ensys_opt.rectifier["parameters"]["efficiency"] / 1000)
        df.loc[0, "diesel_genset_to_demand"] = (ensys_opt.sequences_genset.sum() / 1000
                                                - df.loc[0, "diesel_genset_to_rectifier"])
        df.loc[0, "rectifier_to_dc_bus"] = ensys_opt.sequences_rectifier.sum() / 1000
        df.loc[0, "pv_to_dc_bus"] = ensys_opt.sequences_pv.sum() / 1000
        df.loc[0, "battery_to_dc_bus"] = ensys_opt.sequences_battery_discharge.sum() / 1000
        df.loc[0, "dc_bus_to_battery"] = ensys_opt.sequences_battery_charge.sum() / 1000
        df.loc[0, "dc_bus_to_inverter"] = (ensys_opt.sequences_inverter.sum() /
                                           ensys_opt.inverter["parameters"]["efficiency"] / 1000)
        df.loc[0, "dc_bus_to_surplus"] = ensys_opt.sequences_surplus.sum() / 1000
        df.loc[0, "inverter_to_demand"] = ensys_opt.sequences_inverter.sum() / 1000
        df.loc[0, "time_energy_system_design"] = end_execution_time - start_execution_time
        df.loc[0, "co2_savings"] = co2_savings
        df = df.astype(float).round(3)
        sync_inserts.insert_results_df(df, user_id, project_id)

        # store energy flows
        df = pd.DataFrame()
        df["diesel_genset_production"] = ensys_opt.sequences_genset
        df["pv_production"] = ensys_opt.sequences_pv
        df["battery_charge"] = ensys_opt.sequences_battery_charge
        df["battery_discharge"] = ensys_opt.sequences_battery_discharge
        df["battery_content"] = ensys_opt.sequences_battery_content
        df["demand"] = ensys_opt.sequences_demand
        df["surplus"] = ensys_opt.sequences_surplus
        df = df.round(3)
        sync_inserts.insert_df(models.EnergyFlow, df, user_id, project_id)

        df = pd.DataFrame()
        df["demand"] = ensys_opt.sequences_demand
        df["renewable"] = ensys_opt.sequences_inverter
        df["non_renewable"] = ensys_opt.sequences_genset
        df["surplus"] = ensys_opt.sequences_surplus
        df.index.name = "dt"
        df = df.reset_index()
        df = df.round(3)
        sync_inserts.insert_demand_coverage_df(df, user_id, project_id)

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
        if not ensys_opt.sequences_rectifier.abs().sum() == 0:
            df["rectifier_duration"] = 100 * np.nan_to_num(np.sort(ensys_opt.sequences_rectifier)[::-1]
                                                           / ensys_opt.sequences_rectifier.max())
        else:
            df["rectifier_duration"] = 0
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
        df = df.round(3)
        sync_inserts.insert_df(models.DurationCurve, df, user_id, project_id)
        project_setup = sync_queries.get_model_instance(models.ProjectSetup, user_id, project_id)
        project_setup.status = "finished"
        if project_setup.email_notification is True:
            user = sync_queries.get_user_by_id(user_id)
            subject = "PeopleSun: Model Calculation finished"
            msg = "The calculation of your optimization model is finished. You can view the results at: " \
                  "\n\n{}/simulation_results?project_id={}\n".format(config.DOMAIN, project_id)
            send_mail(user.email, msg, subject=subject)
        project_setup.email_notification = False
        sync_inserts.merge_model(project_setup)
    except Exception as exc:
        user_name = 'user with user_id: {}'.format(user_id)
        error_logger.error_log(exc, 'no request', user_name)
        raise exc


@app.post("/shs_identification/")
def identify_shs(shs_identification_request: fastapi_app.io.schema.ShsIdentificationRequest):
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


# ************************************************************/
# *                     IMPORT / EXPORT                      */
# ************************************************************/

@app.get("/download_data/{project_id}/{file_type}/")
async def export_data(project_id: int, file_type:str,  request: Request):
    user = await accounts.get_user_from_cookie(request)
    input_parameters_df = await queries.get_input_df(user.id, project_id)
    results_df = await queries.get_df(models.Results, user.id, project_id)
    energy_flow_df = await queries.get_df(models.EnergyFlow, user.id, project_id)
    nodes_df = await queries.get_df(models.Nodes, user.id, project_id)
    links_df = await queries.get_df(models.Links, user.id, project_id)
    excel_file = df_to_xlsx(input_parameters_df, energy_flow_df, results_df, nodes_df, links_df)
    response = StreamingResponse(excel_file, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=output.xlsx"
    return response




@app.post("/import_data/{project_id}")
async def import_data(project_id, request: Request, import_files: import_structure = None):
    # add nodes from the 'nodes' sheet of the excel file to the 'nodes.csv' file
    # TODO: update the template for adding nodes
    nodes = import_files["nodes_to_import"]
    links = import_files["links_to_import"]
    user = await accounts.get_user_from_cookie(request)
    user_id = user.id
    if len(nodes) > 0:
        await inserts.update_nodes_and_links(True, False, nodes, user_id, project_id)

    if len(links) > 0:
        await inserts.update_nodes_and_links(False, True, links, user_id, project_id)
