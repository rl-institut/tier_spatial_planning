import json
import asyncio
import decimal
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.sql import text
import flatten_dict
from flatten_dict.splitters import make_splitter
from jose import jwt, JWTError
from sqlalchemy.exc import OperationalError, NoResultFound
from fastapi_app.io.db import models
from fastapi_app.io.db import config
from fastapi_app.io.db.database import get_async_session_maker, async_engine
from fastapi_app.io.db.config import RETRY_COUNT, RETRY_DELAY


async def get_user_by_username(username):
    query =select(models.User).where(models.User.email == username)
    user = await _execute_with_retry(query, which='first')
    return user

async def get_user_by_id(user_id):
    query =select(models.User).where(models.User.id == user_id)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_user_by_guid(guid):
    query =select(models.User).where(models.User.guid == guid)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_max_project_id_of_user(user_id):
    subqry = select(sa.func.max(models.ProjectSetup.project_id)).filter(models.ProjectSetup.id == user_id).as_scalar()
    query = select(models.ProjectSetup).filter(models.ProjectSetup.id == user_id,
                                             models.ProjectSetup.project_id == subqry)
    res = await _execute_with_retry(query, which='first')
    max_project_id = res.project_id if hasattr(res, 'project_id') else None
    return max_project_id


async def next_project_id_of_user(user_id):
    user_id = int(user_id)
    max_project_id = await get_max_project_id_of_user(user_id)
    if pd.isna(max_project_id):
        next_project_id = 0
    else:
        next_project_id = max_project_id + 1
    return next_project_id


async def get_project_of_user(user_id):
    user_id = int(user_id)
    query = select(models.ProjectSetup).where(models.ProjectSetup.id == user_id)
    projects = await _execute_with_retry(query, which='all')
    return projects


async def get_project_setup_of_user(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(models.ProjectSetup).where(models.ProjectSetup.id == user_id,
                                              models.ProjectSetup.project_id == project_id)
    project_setup = await _execute_with_retry(query, which='first')
    return project_setup


async def get_energy_system_design(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(models.EnergySystemDesign).where(models.EnergySystemDesign.id == user_id,
                                                    models.EnergySystemDesign.project_id == project_id)
    model_inst = await _execute_with_retry(query, which='first')
    df = model_inst.to_df()
    if df.empty:
        df = models.Nodes().to_df().iloc[0:0]
    df = df.drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    energy_system_design = flatten_dict.unflatten(df.to_dict('records')[0], splitter=make_splitter('__'))
    return energy_system_design


async def get_links_json(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = await get_df(models.Links, user_id, project_id)
    nodes_json = json.loads(df.to_json())
    return nodes_json


async def get_model_instance(model, user_id, project_id, which='first'):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    model_instance = await _execute_with_retry(query, which=which)
    return model_instance


async def get_input_df(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    project_setup = await get_df(models.ProjectSetup, user_id, project_id, is_timeseries=False)
    grid_design = await get_df(models.GridDesign, user_id, project_id, is_timeseries=False)
    df = pd.concat([project_setup, grid_design], axis=1)
    return df


async def get_df(model, user_id, project_id, is_timeseries=True):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    df = await _get_df(query, is_timeseries=is_timeseries)
    return df

async def _get_df(query, is_timeseries=True):
    if is_timeseries:
        results = await _execute_with_retry(query, which='all')
        if results is not None:
            results = [result.to_dict() for result in results]
        else:
            return None
    else:
        results = await _execute_with_retry(query, which='one')
        if results is not None:
            results = [results.to_dict()]
        else:
            return None
    df = pd.DataFrame.from_records(results)

    if not df.empty:
        if 'id' in df.columns:
            df = df.drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
        decimal_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, decimal.Decimal)).any()]
        for col in decimal_columns:
            df[col] = df[col].astype(float)
    if 'dt' in df.columns:
        df = df.set_index('dt')
    return df

async def get_weather_data(lat, lon, start, end):
    index = pd.date_range(start, end, freq='1H')
    ts_changed = False
    if end > pd.to_datetime('2023-03-01'):
        end = pd.to_datetime('2022-{}-{}'.format(start.month, start.day)) + (end - start)
        start = pd.to_datetime('2022-{}-{}'.format(start.month, start.day))
        ts_changed = True
    model = models.WeatherData
    lats = pd.Series([14.442, 14.192, 13.942, 13.692, 13.442, 13.192, 12.942, 12.692, 12.442,
                      12.192, 11.942, 11.692, 11.442, 11.192, 10.942, 10.692, 10.442, 10.192,
                      9.942, 9.692, 9.442, 9.192, 8.942, 8.692, 8.442, 8.192, 7.942,
                      7.692, 7.442, 7.192, 6.942, 6.692, 6.442, 6.192, 5.942, 5.692,
                      5.442, 5.192, 4.942, 4.692, 4.442, 4.192, 3.942, 3.692, 3.442,
                      3.192, 2.942, 2.691])
    lons = pd.Series([4.24, 4.490026, 4.740053, 4.990079, 5.240105, 5.490131,
                      5.740158, 5.990184, 6.240211, 6.490237, 6.740263, 6.99029,
                      7.240316, 7.490342, 7.740368, 7.990395, 8.240421, 8.490447,
                      8.740474, 8.9905, 9.240526, 9.490553, 9.740579, 9.990605,
                      10.240631, 10.490658, 10.740685, 10.99071, 11.240737, 11.490763,
                      11.740789, 11.990816, 12.240842, 12.490869, 12.740894, 12.990921,
                      13.240948, 13.490973, 13.741])
    closest_lat = round(lats.loc[(lats - lat).abs().idxmin()], 3)
    closest_lon = round(lons.loc[(lons - lon).abs().idxmin()], 3)
    query = select(model).where(model.lat == closest_lat, model.lon == closest_lon, model.dt >= start, model.dt <= end)
    df = await _get_df(query, is_timeseries=True)
    if ts_changed:
        df.index = index
    return df


async def _get_user_from_token(token):
    try:
        payload = jwt.decode(token, config.KEY_FOR_TOKEN, algorithms=[config.TOKEN_ALG])
        username = payload.get("sub")
    except JWTError:
        return None
    # if isinstance(db, scoped_session):
    query = select(models.User).where(models.User.email == username)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_user_from_task_id(task_id):
    query = select(models.User).where(models.User.task_id == task_id)
    user = await _execute_with_retry(query, which='first')
    return user


async def _execute_with_retry(query, which='first'):
    new_engine = False
    for i in range(RETRY_COUNT):
        try:
            async with get_async_session_maker(async_engine, new_engine) as async_db:
                res = await async_db.execute(query)
                if which == 'first':
                    res = res.scalars().first()
                elif which == 'all':
                    res = res.scalars().all()
                elif which == 'one':
                    res = res.scalars().one()
                return res
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < RETRY_COUNT - 1:  # Don't wait after the last try
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"Failed to execute query after {RETRY_COUNT} retries")
                return None
        except NoResultFound:
            return None
