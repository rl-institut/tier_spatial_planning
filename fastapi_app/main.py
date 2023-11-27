import uuid
import ast
import asyncio
import json
import base64
import random
from captcha.image import ImageCaptcha
import pandas as pd
from jose import jwt
import fastapi_app.db.pydantic_schema
from celery_worker import worker
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import Dict
import fastapi_app.helper.identify_consumers_on_map as bi
import fastapi_app.db.sa_tables as models
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from fastapi.staticfiles import StaticFiles
from fastapi_app.models.optimize import check_data_availability, optimize_grid, optimize_energy_system
from fastapi_app.helper.handle_user_accounts import Hasher, create_guid, is_valid_credentials, send_activation_link, activate_mail, \
    authenticate_user, create_access_token, send_mail, create_default_user_account
from fastapi_app.helper import handle_user_accounts as accounts
from fastapi_app.db import async_inserts, async_queries, sync_queries
from fastapi_app import config
from fastapi_app.helper.df_to_excel import df_to_xlsx
import pyutilib.subprocess.GlobalData
from fastapi_app.helper.error_logger import logger as error_logger
from fastapi_app.data.demand.demand_time_series import demand_time_series_df

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

app = FastAPI()
app.mount("/fastapi_app/static", StaticFiles(directory="fastapi_app/static"), name="static")
templates = Jinja2Templates(directory="fastapi_app/pages")
captcha_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@app.on_event("startup")
async def startup_event():
    if not config.DOCKERIZED and not sync_queries.check_if_weather_data_exists():
        pass
        # sync_inserts.update_weather_db()
    await create_default_user_account()


@app.get("/workshop_tasks")
async def get_workshop_slides():
    file_path = "fastapi_app/static/images/workshop.pdf"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf")
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    try:
        user = await accounts.get_user_from_cookie(request)
        user_name = user.email
        projects = await async_queries.get_projects_of_user(user.id)
        for project in projects:
            if project.status == "in progress":
                project.status = "failed"
                await async_inserts.merge_model(project)
                break
    except Exception:
        user_name = 'unknown username'
    error_logger.error_log(exc, request, user_name)
    return RedirectResponse(url="/?internal_error", status_code=303)


@app.post("/renew_token")
async def renew_token(request: Request):
    token = request.cookies.get('access_token', None)
    if token:
        token = token.replace("Bearer ", "")
        token_data = jwt.decode(token, config.KEY_FOR_ACCESS_TOKEN, algorithms=[config.TOKEN_ALG])
        if token_data.get("exp"):
            time_left = token_data.get("exp") - datetime.utcnow().timestamp()
            if time_left < 1200:
                access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
                new_token = create_access_token(data={"sub": token_data.get("sub")}, expires_delta=access_token_expires)
                return {"access_token": new_token}
    return None


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    path = "fastapi_app/static/assets/favicon/favicon.ico"
    return FileResponse(path)


@app.get('/get_captcha')
async def captcha():
    captcha_text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))
    captcha = ImageCaptcha()
    loop = asyncio.get_running_loop()
    captcha_data = await loop.run_in_executor(None, captcha.generate, captcha_text)
    base64_image = base64.b64encode(captcha_data.getvalue()).decode('utf-8')
    hashed_captcha = captcha_context.hash(captcha_text)
    return JSONResponse({'img': base64_image, 'hashed_captcha': hashed_captcha})


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
        projects = await async_queries.get_projects_of_user(user.id)
        for project in projects:
            project.created_at = project.created_at.date()
            project.updated_at = project.updated_at.date()
            if user.task_id is not None and project.project_id == user.project_id:
                if bool(os.environ.get('DOCKERIZED')):
                    status = worker.AsyncResult(user.task_id).status.lower()
                else:
                    status = 'success'
                if status in ['success', 'failure', 'revoked']:
                    project_setup = await async_queries.get_model_instance(models.ProjectSetup, user.id, user.project_id)
                    user.task_id = ''
                    user.project_id = None
                    await async_inserts.update_model_by_user_id(user)
                    if status == 'success':
                        project_setup.status = "finished"
                    else:
                        project_setup.status = status
                    await async_inserts.merge_model(project_setup)
                    user.task_id = ''
                    user.project_id = None
                    await async_inserts.update_model_by_user_id(user)
        return templates.TemplateResponse("user_projects.html", {"request": request,
                                                                 'projects': projects})


@app.get("/project_setup", response_class=HTMLResponse)
async def project_setup(request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_id = request.query_params.get('project_id')
    if project_id is None:
        project_id = await async_queries.next_project_id_of_user(user.id)
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
        user = await async_queries.get_user_by_guid(guid)
        if user is not None:
            return templates.TemplateResponse("reset_password.html", {"request": request, 'guid': guid})
    return RedirectResponse('/')


@app.post("/reset_password")
async def reset_password(form_data: Dict[str, str]):
    guid = form_data.get('guid')
    password = form_data.get('password')
    if guid is not None:
        user = await async_queries.get_user_by_guid(guid)
        if user is not None:
            if accounts.is_valid_password(password):
                user.hashed_password = Hasher.get_password_hash(password)
                user.guid = ''
                await async_inserts.merge_model(user)
                res = 'Password changed successfully.'
                validation = True
            else:
                validation = False
                res = 'The password needs to be at least 8 characters long'
            return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.get("/account_overview")
async def account_overview(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user is None or 'anonymous__' in user.email:
        return RedirectResponse('/')
    else:
        return templates.TemplateResponse("account_overview.html", {"request": request,
                                                                    'email': user.email})

@app.get("/contact")
async def contact(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user is None or 'anonymous__' in user.email:
        email = ''
    else:
        email = user.email
    return templates.TemplateResponse("contact_form.html", {"request": request, 'email': email})


@app.get("/example_model")
async def example_model(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if user is not None:
        await async_inserts.insert_example_project(user.id)
    return JSONResponse(status_code=200, content={'success': True})


@app.get("/copy_project")
async def copy_project(request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_id = request.query_params.get('project_id')
    if user is not None and project_id is not None:
        await async_inserts.copy_project(user.id, project_id)
        return JSONResponse(status_code=200, content={'success': True})
    else:
        return JSONResponse(status_code=400, content={'success': False})

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
        await async_inserts.remove_project(user.id, project_id)


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
                                                               'time': 3,
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
    project_setup = await async_queries.get_model_instance(models.ProjectSetup, user.id, project_id)
    project_setup.email_notification = is_active
    await async_inserts.merge_model(project_setup)


@app.get("/db_links_to_js/{project_id}")
async def db_links_to_js(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    links = await async_queries.get_model_instance(models.Links, user.id, project_id)
    links_json = json.loads(links.data) if links is not None else json.loads('{}')
    return JSONResponse(content=links_json, status_code=200)


@app.get("/db_nodes_to_js/{project_id}/{markers_only}")
async def db_nodes_to_js(project_id: str, markers_only: bool, request: Request):
    user = await accounts.get_user_from_cookie(request)
    if project_id == 'undefined':
        project_id = get_project_id_from_request(request)
    nodes = await async_queries.get_model_instance(models.Nodes, user.id, project_id)
    df = pd.read_json(nodes.data) if nodes is not None else pd.DataFrame()
    if not df.empty:
        df = df[['latitude',
                 'longitude',
                 'how_added',
                 'node_type',
                 'consumer_type',
                 'consumer_detail',
                 'custom_specification',
                 'is_connected',
                 'shs_options']]
        power_house = df[df['node_type'] == 'power-house']
        if markers_only is True:
            if len(power_house) > 0 and power_house['how_added'].iat[0] == 'manual':
                df = df[df['node_type'].isin(['power-house', 'consumer'])]
            else:
                df = df[df['node_type'] == 'consumer']
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        df['shs_options'] = df['shs_options'].fillna(0)
        df['custom_specification'] = df['custom_specification'].fillna('')
        df['shs_options'] = df['shs_options'].astype(int)
        df['is_connected'] = df['is_connected'].astype(bool)
        nodes_list = df.to_dict('records')
        is_load_center = True
        if len(power_house.index) > 0 and power_house['how_added'].iat[0] =='manual':
            is_load_center = False
        return JSONResponse(status_code=200, content={'is_load_center': is_load_center, "map_elements": nodes_list})
    else:
        return None


@app.post("/consumer_to_db/{project_id}")
async def consumer_to_db(project_id: str, map_elements: fastapi_app.db.pydantic_schema.MapDataRequest, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = pd.DataFrame.from_records(map_elements.map_elements)
    if df.empty is True:
        await async_inserts.remove(models.Nodes, user.id, project_id)
        return
    df = df.drop_duplicates(subset=['latitude', 'longitude'])
    drop_index = df[df['node_type'] == 'power-house'].index
    if drop_index.__len__() > 1:
        df = df.drop(index=drop_index[1:])
    if df.empty is True:
        await async_inserts.remove(models.Nodes, user.id, project_id)
        return
    df = df[df['node_type'].isin(['power-house', 'consumer'])]
    if df.empty is True:
        await async_inserts.remove(models.Nodes, user.id, project_id)
        return
    df = df[['latitude', 'longitude', 'how_added', 'node_type', 'consumer_type', 'custom_specification', 'shs_options',
             'consumer_detail']]
    df['consumer_type'] = df['consumer_type'].fillna('household')
    df['custom_specification'] = df['custom_specification'].fillna('')
    df['shs_options'] = df['shs_options'].fillna(0)
    df['is_connected'] = True
    df = df.round(decimals=6)
    if df.empty:
        await async_inserts.remove(models.Nodes, user.id, project_id)
        return
    df["node_type"] = df["node_type"].astype(str)
    if len(df.index) != 0:
        if 'parent' in df.columns:
            df['parent'] = df['parent'].replace('unknown', None)
    nodes = models.Nodes()
    nodes.id = user.id
    nodes.project_id = project_id
    nodes.data = df.reset_index(drop=True).to_json()
    await async_inserts.merge_model(nodes)
    return JSONResponse(status_code=200, content={"message": "Success"})


@app.get("/load_results/{project_id}")
async def load_results(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await async_queries.get_df(models.Results, user.id, project_id)
    infeasible = bool(df.loc[0, 'infeasible']) if df.columns.__contains__('infeasible') else False
    if df.empty:
        await asyncio.sleep(1)
        df = await async_queries.get_df(models.Results, user.id, project_id)
        if df.empty or df['lcoe'].isna() is True:
            return JSONResponse(content={})
    if bool(df['lcoe'].isna()[0]) is True:
        return JSONResponse(content={})
    df["average_length_distribution_cable"] = df["length_distribution_cable"] / df["n_distribution_links"]
    df["average_length_connection_cable"] = df["length_connection_cable"] / df["n_connection_links"]
    df["time"] = (df["time_grid_design"] + df["time_energy_system_design"]) * 3
    df["gridLcoe"] = df['cost_grid'].astype(float) / df["epc_total"].astype(float) * 100
    df["esLcoe"] = (df["epc_total"].astype(float) - df['cost_grid'].astype(float)) \
                   / df["epc_total"].astype(float) * 100
    unit_dict = {'n_poles': '',
                 'n_consumers': '',
                 'n_shs_consumers': '',
                 'length_distribution_cable': 'm',
                 'average_length_distribution_cable': 'm',
                 'length_connection_cable': 'm',
                 'average_length_connection_cable': 'm',
                 'cost_grid': 'USD/a',
                 'lcoe': '',
                 'gridLcoe': '%',
                 'esLcoe': '%',
                 'res': '%',
                 'max_voltage_drop': '%',
                 'shortage_total': '%',
                 'surplus_rate': '%',
                 'time': 's',
                 'co2_savings': 't/a',
                 'total_annual_consumption': 'kWh/a',
                 'average_annual_demand_per_consumer': 'W',
                 'upfront_invest_grid': 'USD',
                 'upfront_invest_diesel_gen': 'USD',
                 'upfront_invest_inverter': 'USD',
                 'upfront_invest_rectifier': 'USD',
                 'upfront_invest_battery': 'USD',
                 'upfront_invest_pv': 'USD',
                 'upfront_invest_converters': 'USD',
                 'upfront_invest_total': 'USD',
                 'battery_capacity': 'kWh',
                 'pv_capacity': 'kW',
                 'diesel_genset_capacity': 'kW',
                 'inverter_capacity': 'kW',
                 'rectifier_capacity': 'kW',
                 'co2_emissions': 't/a',
                 'fuel_consumption': 'kWh',
                 'peak_demand': 'kW',
                 'base_load': 'kW',
                 'max_shortage': '%',
                 'cost_fuel': 'USD/a',
                 'epc_pv': 'USD/a',
                 'epc_diesel_genset': 'USD/a',
                 'epc_inverter': 'USD/a',
                 'epc_rectifier': 'USD/a',
                 'epc_battery': 'USD/a',
                 'epc_total': 'USD/a'
                 }
    if int(df['n_consumers'].iat[0]) != int(df['n_shs_consumers'].iat[0]) and not infeasible:
        df['upfront_invest_converters'] = sum(df[col].iat[0] for col in df.columns if 'upfront' in col and 'grid' not in col)
        df['upfront_invest_total'] = df['upfront_invest_converters'] + df['upfront_invest_grid']
    else:
        df['upfront_invest_converters'] = None
        df['upfront_invest_total'] = None
    df = df[list(unit_dict.keys())].round(1).astype(str)
    for col in df.columns:
        if unit_dict[col] in ['%', 's', 'kW', 'kWh']:
            df[col] = df[col].where(df[col] != 'None', 0)
            if df[col].isna().sum() == 0:
                df[col] = df[col].astype(float).round(1).astype(str)
        elif unit_dict[col] in ['USD', 'kWh/a', 'USD/a']:
            if df[col].isna().sum() == 0 and df.loc[0, col] != 'None':
                df[col] = "{:,}".format(df[col].astype(float).astype(int).iat[0])
        df[col] = df[col] + ' ' + unit_dict[col]
    results = df.to_dict(orient='records')[0]
    if infeasible is True:
        results['responseMsg'] = 'There are no results of the energy system optimization. There were no feasible ' \
                                 'solution.'
    elif int(results['n_consumers']) == int(results['n_shs_consumers']):
        results['responseMsg'] = 'Due to high grid costs, all consumers have been equipped with solar home ' \
                                 'systems. A grid was not built, therefore no optimization of the energy system was ' \
                                 'carried out.'
    else:
        results['responseMsg'] = ''
    return JSONResponse(content=results, status_code=200)



@app.get("/show_video_tutorial")
async def show_video_tutorial(request: Request):
    user = await accounts.get_user_from_cookie(request)
    if pd.isna(user.show_tutorial):
        show_tutorial = True
    else:
        show_tutorial = bool(user.show_tutorial)
    return show_tutorial


@app.get("/deactivate_video_tutorial")
async def deactivate_video_tutorial(request: Request):
    user = await accounts.get_user_from_cookie(request)
    user.show_tutorial = False
    await async_inserts.merge_model(user)


@app.get("/load_previous_data/{page_name}")
async def load_previous_data(page_name, request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_id = request.query_params.get('project_id')
    if page_name == "project_setup":
        if project_id == 'new':
            project_id = await async_queries.next_project_id_of_user(user.id)
            return models.ProjectSetup(project_id=project_id)
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        project_setup = await async_queries.get_model_instance(models.ProjectSetup, user.id, project_id)
        if hasattr(project_setup, 'start_date'):
            project_setup.start_date = project_setup.start_date.date().__str__()
            return project_setup
        else:
            return None
    elif page_name == "grid_design":
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        grid_design = await async_queries.get_model_instance(models.GridDesign, user.id, project_id)
        return grid_design
    elif page_name == "demand_estimation":
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        demand_estimation = await async_queries.get_model_instance(models.Demand, user.id, project_id)
        if demand_estimation is None or not hasattr(demand_estimation,'maximum_peak_load'):
            return None
        demand_estimation.maximum_peak_load = str(demand_estimation.maximum_peak_load) \
            if demand_estimation.maximum_peak_load is not None else ''
        demand_estimation.average_daily_energy = str(demand_estimation.average_daily_energy) \
            if demand_estimation.average_daily_energy is not None else ''
        demand_estimation.custom_calibration = True \
            if len(demand_estimation.maximum_peak_load) > 0 or len(demand_estimation.average_daily_energy) > 0 \
            else False
        demand_estimation.calibration_options = 2 if len(demand_estimation.maximum_peak_load) > 0 else 1
        return demand_estimation
    elif page_name == 'energy_system_design':
        try:
            project_id = int(project_id)
        except (ValueError, TypeError):
            return None
        energy_system_design = await async_queries.get_model_instance(models.EnergySystemDesign, user.id, project_id)
        return energy_system_design


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
            send_activation_link(email, guid)
            user = models.User(email=email,
                               hashed_password=Hasher.get_password_hash(password),
                               guid=guid,
                               is_confirmed=False,
                               is_active=False,
                               is_superuser=False)
            await async_inserts.merge_model(user)
        else:
            res = [False, 'Please enter a valid captcha']
    return fastapi_app.db.pydantic_schema.ValidRegistration(validation=res[0], msg=res[1])


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
        await async_inserts.merge_model(user)
        user = await async_queries.get_user_by_username(name)
        access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS)
        access_token = create_access_token(data={"sub": name}, expires_delta=access_token_expires)
        response.set_cookie(key="access_token", value=f"Bearer {access_token}",
                            httponly=True)  # set HttpOnly cookie in response
        if bool(os.environ.get('DOCKERIZED')):
            minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES_ANONYMOUS + 60
            eta = datetime.utcnow() + timedelta(minutes=minutes)
            task_remove_anonymous_users.apply_async((user.id,), eta=eta)
        validation, res = True, ''
    else:
        validation, res = False, 'Please enter a valid captcha'
    return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.post("/login/")
async def login(response: Response, credentials: fastapi_app.db.pydantic_schema.Credentials):
    if isinstance(credentials.email, str) and len(credentials.email) > 3:
        is_valid, res = await authenticate_user(credentials.email.strip(), credentials.password)
        if is_valid:
            if credentials.remember_me:
                access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES_EXTENDED)
            else:
                access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
            del credentials
            access_token = create_access_token(data={"sub": res.email}, expires_delta=access_token_expires)
            response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
            return fastapi_app.db.pydantic_schema.ValidRegistration(validation=True, msg="")
        else:
            del credentials
            return fastapi_app.db.pydantic_schema.ValidRegistration(validation=False, msg=res)


@app.post("/consent_cookie/")
async def consent_cookie(response: Response):
    expire_date = datetime.utcnow()
    expire_date = expire_date + timedelta(days=365)
    expire_date = expire_date.replace(tzinfo=timezone.utc)
    response.set_cookie(key="consent_cookie", value='true', httponly=True, expires=expire_date)


@app.post("/change_email/")
async def change_email(request: Request, credentials: fastapi_app.db.pydantic_schema.Credentials):
    if isinstance(credentials.email.strip(), str) and len(credentials.email.strip()) > 3:
        user = await accounts.get_user_from_cookie(request)
        is_valid, res = await authenticate_user(user.email, credentials.password)
        validation = False
        if is_valid:
            user.email = credentials.email.strip()
            if accounts.is_valid_email(user):
                user.is_confirmed = False
                user.guid = create_guid()
                await async_inserts.merge_model(user)
                send_activation_link(credentials.email.strip(), user.guid)
                res = 'Please click the activation link we sent to your email.'
                validation = True
            else:
                res = 'Please enter a valid email address.'
        else:
            del credentials
        return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.post("/change_pw/")
async def change_pw(request: Request, passwords: Dict[str, str]):
    user = await accounts.get_user_from_cookie(request)
    is_valid, res = await authenticate_user(user.email, passwords['old_password'])
    validation = False
    if is_valid:
        if accounts.is_valid_password(passwords['new_password']):
            user.hashed_password = Hasher.get_password_hash(passwords['new_password'])
            await async_inserts.merge_model(user)
            res = 'Password changed successfully.'
            validation = True
        else:
            res = 'The password needs to be at least 8 characters long'
    else:
        del passwords
    return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.post("/send_reset_password_email/")
async def send_reset_password_email(data: Dict[str, str]):
    email = data.get('email').strip()
    captcha_input = data.get('captcha_input')
    hashed_captcha = data.get('hashed_captcha')
    user = await async_queries.get_user_by_username(email)
    if user is None:
        validation, res = False, 'Email address is not registered'
    else:
        if captcha_context.verify(captcha_input, hashed_captcha):
            guid = str(uuid.uuid4()).replace('-', '')[:24]
            user.guid = guid
            await async_inserts.merge_model(user)
            msg = 'For your PeopleSuN-Account a password reset was requested. If you did not request a password reset, ' \
                  'please ignore this email. Otherwise, please click the following link:\n\n{}/reset_password?guid={}' \
                .format(config.DOMAIN, guid)
            send_mail(email, msg, 'PeopleSuN-Account: Reset your password')
            validation, res = True, 'Please click the link we sent to your email.'
        else:
            validation, res = False, 'Please enter a valid captcha'
    return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.post("/delete_account/")
async def change_pw(response: Response, request: Request, form_data: fastapi_app.db.pydantic_schema.Password):
    user = await accounts.get_user_from_cookie(request)
    is_valid, res = await authenticate_user(user.email, form_data.password)
    validation = False
    if is_valid:
        await async_inserts.remove_account(user.id)
        response.delete_cookie("access_token")
        res = 'Account removed'
        validation = True
    return fastapi_app.db.pydantic_schema.ValidRegistration(validation=validation, msg=res)


@app.post("/logout/")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"status": "success"}


@app.post("/query_account_data/")
async def query_account_data(project_id: fastapi_app.db.pydantic_schema.ProjectID, request: Request):
    user = await accounts.get_user_from_cookie(request)
    project_name = ''
    if user is not None:
        name = user.email
        if 'anonymous__' in name:
            name = name.split('__')[0]
        else:
            if project_id.project_id is not None:
                project = await async_queries.get_project_name_by_id(user.id, project_id)
                project_name = project.project_name
        return fastapi_app.db.pydantic_schema.UserOverview(email=name, project_name=project_name)
    else:
        return fastapi_app.db.pydantic_schema.UserOverview(email="", project_name="")


@app.post("/has_cookie/")
async def has_cookie(request: Request, has_cookies: fastapi_app.db.pydantic_schema.HasCookies):
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
async def save_grid_design(request: Request, data: fastapi_app.db.pydantic_schema.SaveGridDesign):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    data.grid_design['id'] = user.id
    data.grid_design['project_id'] = project_id
    grid_design = models.GridDesign(**data.grid_design)
    await async_inserts.merge_model(grid_design)
    return JSONResponse(status_code=200, content={"message": "Success"})


@app.post("/save_demand_estimation/")
async def save_demand_estimation(request: Request, data: fastapi_app.db.pydantic_schema.SaveDemandEstimation):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    custom_calibration = ast.literal_eval(data.demand_estimation['custom_calibration'])
    use_custom_shares_bool = ast.literal_eval(data.demand_estimation['use_custom_shares'])
    use_custom_shares = 0
    custom_share_1, custom_share_2, custom_share_3, custom_share_4, custom_share_5 = 0, 0, 0, 0, 0

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

    if use_custom_shares_bool is None or '':
        use_custom_shares = 0
    else:
        try:
            if use_custom_shares_bool:
                use_custom_shares = 1
                custom_share_1 = round(float(data.demand_estimation['custom_share_1']), 1)
                custom_share_2 = round(float(data.demand_estimation['custom_share_2']), 1)
                custom_share_3 = round(float(data.demand_estimation['custom_share_3']), 1)
                custom_share_4 = round(float(data.demand_estimation['custom_share_4']), 1)
                custom_share_5 = round(float(data.demand_estimation['custom_share_5']), 1)

        except ValueError:
            use_custom_shares = 0

    dictionary = {'id': user.id,
                  'project_id': project_id,
                  'household_option': data.demand_estimation['household_option'],
                  'maximum_peak_load': maximum_peak_load,
                  'average_daily_energy': average_daily_energy,
                  'use_custom_shares': use_custom_shares,
                  'custom_share_1': custom_share_1,
                  'custom_share_2': custom_share_2,
                  'custom_share_3': custom_share_3,
                  'custom_share_4': custom_share_4,
                  'custom_share_5': custom_share_5,}
    demand_estimation = models.Demand(**dictionary)
    await async_inserts.merge_model(demand_estimation)
    return JSONResponse(status_code=200, content={"message": "Success"})


@app.post("/send_mail_route/")
async def send_mail_route(mail: fastapi_app.db.pydantic_schema.Mail):
    body = 'offgridplanner.org contact form. email from: {}'.format(mail.from_address) + '\n' + mail.body
    subject = 'offgridplanner.org contact form: {}'.format(mail.subject)
    send_mail('internal', body, subject)
    return JSONResponse(status_code=200, content={"message": "Success"})


@app.post("/save_project_setup/{project_id}")
async def save_project_setup(project_id, request: Request, data: fastapi_app.db.pydantic_schema.SaveProjectSetup):
    user = await accounts.get_user_from_cookie(request)
    # project_id = get_project_id_from_request(request)
    timestamp = pd.Timestamp.now()
    data.page_setup['created_at'] = timestamp
    data.page_setup['updated_at'] = timestamp
    data.page_setup['id'] = user.id
    data.page_setup['project_id'] = project_id
    project_setup = models.ProjectSetup(**data.page_setup)
    await async_inserts.merge_model(project_setup)
    return JSONResponse(status_code=200, content={"message": "Success"})


@app.post("/save_energy_system_design/")
async def save_energy_system_design(request: Request, data: fastapi_app.db.pydantic_schema.OptimizeEnergySystemRequest):
    user = await accounts.get_user_from_cookie(request)
    project_id = get_project_id_from_request(request)
    df = data.to_df()
    await async_inserts.insert_energysystemdesign_df(df, user.id, project_id)


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


@app.get("/get_plot_data/{project_id}")
async def get_plot_data(project_id, request: Request):
    user = await accounts.get_user_from_cookie(request)
    df = await async_queries.get_df(models.Results, user.id, project_id)
    df = df.astype(str)
    optimal_capacity_keys = ["pv", "battery", "inverter", "rectifier", "diesel_genset", "peak_demand", "surplus"]
    optimal_capacities = {key: df.loc[0, f"{key}_capacity"] for key in optimal_capacity_keys[:-2]}
    optimal_capacities.update({key: df.loc[0, key] for key in optimal_capacity_keys[-2:]})
    lcoe_breakdown_keys = ["renewable_assets", "non_renewable_assets", "grid", "fuel"]
    lcoe_breakdown = {key: df.loc[0, f"cost_{key}"] for key in lcoe_breakdown_keys}
    sankey_keys = ["fuel_to_diesel_genset", "diesel_genset_to_rectifier", "diesel_genset_to_demand",
                   "rectifier_to_dc_bus", "pv_to_dc_bus", "battery_to_dc_bus", "dc_bus_to_battery",
                   "dc_bus_to_inverter", "dc_bus_to_surplus", "inverter_to_demand"]
    sankey_data = {key: df.loc[0, key] for key in sankey_keys}
    energy_flow = await async_queries.get_model_instance(models.EnergyFlow, user.id, project_id)
    energy_flow = pd.read_json(energy_flow.data)
    energy_flow['battery'] = energy_flow['battery_discharge'] - energy_flow['battery_charge']
    energy_flow.drop(columns=['battery_charge', 'battery_discharge'], inplace=True)
    energy_flow.reset_index(drop=True, inplace=True)
    energy_flow = json.loads(energy_flow.to_json())
    duration_curve = json.loads((await async_queries.get_model_instance(models.DurationCurve, user.id, project_id)).data)
    emissions = json.loads((await async_queries.get_model_instance(models.Emissions, user.id, project_id)).data)
    demand_coverage = json.loads((await async_queries.get_model_instance(models.DemandCoverage, user.id, project_id)).data)
    return JSONResponse(status_code=200, content={"optimal_capacities": optimal_capacities,
                                                  "lcoe_breakdown": lcoe_breakdown,
                                                  "sankey_data": sankey_data,
                                                  "energy_flow": energy_flow,
                                                  "duration_curve": duration_curve,
                                                  "demand_coverage": demand_coverage,
                                                  "emissions": emissions})


@app.get("/get_demand_time_series/{project_id}")
async def get_demand_time_series(project_id):
    return demand_time_series_df().to_dict('list')  # converts dataframe to dict format with lists as values


@app.post("/add_buildings_inside_boundary")
async def add_buildings_inside_boundary(js_data: fastapi_app.db.pydantic_schema.MapData, request: Request):
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
    data, building_coordidates_within_boundaries = bi.get_consumer_within_boundaries(df)
    if building_coordidates_within_boundaries is None:
        return JSONResponse({'executed': False, 'msg': 'In the selected area, no buildings could be identified.'})
    nodes = defaultdict(list)
    for label, coordinates in building_coordidates_within_boundaries.items():
        nodes["latitude"].append(round(coordinates[0], 6))
        nodes["longitude"].append(round(coordinates[1], 6))
        nodes["how_added"].append("automatic")
        nodes["node_type"].append("consumer")
        nodes["consumer_type"].append('household')
        nodes["consumer_detail"].append('default')
        nodes['custom_specification'].append('')
        nodes['shs_options'].append(0)
        nodes['is_connected'].append(True)
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
    df['is_connected'] = df['is_connected']
    df_exisiting = pd.DataFrame.from_records(js_data.map_elements)
    df = pd.concat([df_exisiting, df], ignore_index=True)
    df = df.drop_duplicates(subset=['longitude', 'latitude'], keep='first')
    df['shs_options'] = df['shs_options'].fillna(0)
    df['custom_specification'] = df['custom_specification'].fillna('')
    df['is_connected'] = df['is_connected'].fillna(True)
    nodes_list = df.to_dict('records')
    return JSONResponse({'executed': True, 'msg': '', 'new_consumers': nodes_list})

@app.post("/remove_buildings_inside_boundary")
async def remove_buildings_inside_boundary(data: fastapi_app.db.pydantic_schema.MapData):
    df = pd.DataFrame.from_records(data.map_elements)
    if not df.empty:
        boundaries = pd.DataFrame.from_records(data.boundary_coordinates[0][0]).values.tolist()
        df['inside_boundary'] = bi.are_points_in_boundaries(df, boundaries=boundaries, )
        df = df[df['inside_boundary'] == False]
        df = df.drop(columns=['inside_boundary'])
        return JSONResponse({'map_elements': df.to_dict('records')})


# add new manually-selected nodes to the *.csv file
@app.post("/database_add_remove_manual/{add_remove}/{project_id}")
async def database_add_remove_manual(add_remove: str, project_id, add_node_request: fastapi_app.db.pydantic_schema.AddNodeRequest,
                                     request: Request):
    user = await accounts.get_user_from_cookie(request)
    nodes = models.Nodes(**dict(add_node_request)).to_dict()
    if add_remove == "remove":
        nodes = await async_queries.get_model_instance(models.Nodes, user.id, project_id)
        df = pd.read_json(nodes.data)
        df = df[(df["node_type"] != "pole") & (df["node_type"] != "power-house")]
        for index in df.index:
            if (round(add_node_request.latitude, 6) == df.to_dict()["latitude"][index]) and \
                    (round(add_node_request.longitude, 6) == df.to_dict()["longitude"][index]):
                df.drop(labels=index, axis=0, inplace=True)
        df = df.reset_index(drop=True)
        await async_inserts.update_nodes_and_links(True, False, df.to_dict(), user.id, project_id, add=False)
    else:
        await async_inserts.update_nodes_and_links(True, False, nodes, user.id, project_id, add=True, replace=False)


async def remove_results(user_id, project_id):
    await async_inserts.remove(models.Results, user_id, project_id)
    await async_inserts.remove(models.DemandCoverage, user_id, project_id)
    await async_inserts.remove(models.EnergyFlow, user_id, project_id)
    await async_inserts.remove(models.Emissions, user_id, project_id)
    await async_inserts.remove(models.DurationCurve, user_id, project_id)
    await async_inserts.remove(models.Links, user_id, project_id)


@worker.task(name='celery_worker.task_grid_opt',
             force=True,
             track_started=True,
             autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 1, 'countdown': 10})
def task_grid_opt(user_id, project_id):
    result = optimize_grid(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_supply_opt',
             force=True,
             track_started=True,
             autoretry_for=(Exception,),
             retry_kwargs={'max_retries': 1, 'countdown': 10}
             )
def task_supply_opt(user_id, project_id):
    result = optimize_energy_system(user_id, project_id)
    return result


@worker.task(name='celery_worker.task_remove_anonymous_users', force=True, track_started=True)
def task_remove_anonymous_users(user_id):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        result = loop.run_until_complete(async_inserts.remove_account(user_id))
    else:
        result = asyncio.run(async_inserts.remove_account(user_id))
    return result

async def optimization(user_id, project_id):
    await remove_results(user_id, project_id)
    project_setup = await async_queries.get_model_instance(models.ProjectSetup, user_id, project_id)
    project_setup.status = "queued"
    await async_inserts.merge_model(project_setup)
    if bool(os.environ.get('DOCKERIZED')):
        task = task_grid_opt.delay(user_id, project_id)
        return task.id
    else:
        optimize_grid(user_id, project_id)
        optimize_energy_system(user_id, project_id)
        return 'no_celery_id'


@app.get("/optimize_without_celery/{project_id}")
async def forward_if_consumer_selection_exists(project_id: int, request: Request):

    user = await accounts.get_user_from_cookie(request)
    if bool(user.is_superuser) is True:
        optimize_grid(user.id, project_id)
        optimize_energy_system(user.id, project_id)
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
    nodes = await async_queries.get_model_instance(models.Nodes, user.id, project_id)
    if nodes is None:
        res = {'forward': False}
    else:
        df = pd.read_json(nodes.data)
        if df is not None and len(df['consumer_type']) > 0:
            res = {'forward': True}
        else:
            res = {'forward': False}
    return JSONResponse(res)


@app.post("/start_calculation/{project_id}")
async def start_calculation(project_id, request: Request):
    if project_id is None:
        project_id = request.query_params.get('project_id')
    user = await accounts.get_user_from_cookie(request)
    forward, redirect = await check_data_availability(user.id, project_id)
    if forward is False:
        return JSONResponse({'task_id': '', 'redirect': redirect})
    task_id = await optimization(user.id, project_id)
    user.task_id = task_id
    user.project_id = int(project_id)
    await async_inserts.update_model_by_user_id(user)
    return JSONResponse({'task_id': task_id, 'redirect': ''})


@app.post('/waiting_for_results/')
async def waiting_for_results(request: Request, data: fastapi_app.db.pydantic_schema.TaskInfo):
    async def pause_until_results_are_available(user_id, project_id, status):
        n_iter = 4 if status == 'unknown' else 2
        for i in range(n_iter):
            results = await async_queries.get_model_instance(models.Results, user_id, project_id)
            if hasattr(results, 'lcoe') and results.lcoe is not None:
                break
            elif hasattr(results, 'infeasible') and bool(results.infeasible) is True:
                break
            elif hasattr(results, 'n_consumers') and hasattr(results, 'n_shs_consumers') and \
                results.n_consumers == results.n_shs_consumers:
                break
            else:
                await asyncio.sleep(5 + i)
                print('Results are not available')
    try:
        max_time = 3600 * 24 * 7
        t_wait = -2E-05 *  data.time + 0.0655 *  data.time + 5.7036 if data.time < 1800 else 60
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
            for i in range(4):
                user = await accounts.get_user_from_cookie(request)
                if user is not None:
                    break
                else:
                    print('Could not get user from cookie')
                await asyncio.sleep(1)
                if i == 2 and len(data.task_id) > 12 and max_time > res['time']:
                    user = await async_queries.get_user_from_task_id(data.task_id)
                    if user is not None:
                        break
                    else:
                        print('Could not get user from task id')
            if data.model == 'grid' and bool(os.environ.get('DOCKERIZED')):
                task = task_supply_opt.delay(user.id, data.project_id)
                user.task_id = task.id
                await async_inserts.update_model_by_user_id(user)
                res['finished'] = False
                res['status'] = "power supply optimization is running..."
                res['model'] = 'supply'
                res[task.id] = task.id
            else:
                project_setup = await async_queries.get_model_instance(models.ProjectSetup, user.id, data.project_id)
                if project_setup is not None:
                    if 'status' in locals():
                        if status in ['success', 'failure', 'revoked']:
                            project_setup.status = "finished"
                        else:
                            project_setup.status = status
                    else:
                        project_setup.status = "finished"
                        status = 'unknown'
                    await async_inserts.merge_model(project_setup)
                    user.task_id = ''
                    user.project_id = None
                    await async_inserts.update_model_by_user_id(user)
                    await pause_until_results_are_available(user.id, data.project_id, status)
        return JSONResponse(res)
    except Exception as e:
        print(e)
        user_name = user.username if hasattr(user, 'username') else 'unknown'
        error_logger.error_log(e, request, user_name)
        raise Exception(e)


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
async def revoke_task(request: Request, data: fastapi_app.db.pydantic_schema.TaskID):
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
    await async_inserts.update_model_by_user_id(user)


@app.get("/download_data/{project_id}/{file_type}/")
async def export_data(project_id: int, file_type:str,  request: Request):
    user = await accounts.get_user_from_cookie(request)
    input_parameters_df = await async_queries.get_input_df(user.id, project_id)
    results_df = await async_queries.get_df(models.Results, user.id, project_id)
    energy_flow = await async_queries.get_model_instance(models.EnergyFlow, user.id, project_id)
    energy_flow_df = pd.read_json(energy_flow.data) if energy_flow is not None else pd.DataFrame()
    nodes = await async_queries.get_model_instance(models.Nodes, user.id, project_id)
    links = await async_queries.get_model_instance(models.Links, user.id, project_id)
    nodes_df = pd.read_json(nodes.data) if nodes is not None else pd.DataFrame()
    links_df = pd.read_json(links.data) if links is not None else pd.DataFrame()
    energy_system_design = await async_queries.get_df(models.EnergySystemDesign, user.id, project_id)
    excel_file = df_to_xlsx(input_parameters_df, energy_system_design, energy_flow_df, results_df, nodes_df, links_df)
    response = StreamingResponse(excel_file, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    response.headers["Content-Disposition"] = "attachment; filename=offgridplanner_results.xlsx"
    return response
