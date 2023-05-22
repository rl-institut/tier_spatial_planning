import json
import decimal
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.sql import text
import flatten_dict
from flatten_dict.splitters import make_splitter
from fastapi_app.io.db import models
from fastapi_app.io.db.database import get_async_session_maker, get_sync_session_maker


async def get_user_by_username(username):
    query =select(models.User).where(models.User.email == username)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        user = res.scalars().first()
    return user

async def get_user_by_id(user_id):
    query =select(models.User).where(models.User.id == user_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        user = res.scalars().first()
    return user


async def get_user_by_guid(guid):
    query =select(models.User).where(models.User.guid == guid)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        user = res.scalars().first()
    return user


async def get_max_project_id_of_user(user_id):
    subqry = select(sa.func.max(models.ProjectSetup.project_id)).filter(models.ProjectSetup.id == user_id).as_scalar()
    qry = select(models.ProjectSetup).filter(models.ProjectSetup.id == user_id,
                                             models.ProjectSetup.project_id == subqry)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(qry)
    res = res.scalars().first()
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
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(select(models.ProjectSetup).where(models.ProjectSetup.id == user_id))
        projects = res.scalars().all()
    return projects


async def get_project_setup_of_user(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(models.ProjectSetup).where(models.ProjectSetup.id == user_id,
                                              models.ProjectSetup.project_id == project_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        project_setup = res.scalars().first()
    return project_setup


async def get_energy_system_design(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(models.EnergySystemDesign).where(models.EnergySystemDesign.id == user_id,
                                                    models.EnergySystemDesign.project_id == project_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        model_inst = res.scalars().first()
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


async def get_model_instance(model, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        model_instance = res.scalars().first()
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
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
    if is_timeseries:
        results = res.scalars().all()
        results = [result.to_dict() for result in results]
    else:
        results = [res.scalars().one().to_dict()]
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
    if end > pd.to_datetime('2023-05-01'):
        start = pd.to_datetime('2022-{}-{}'.format(start.month, start.day))
        end = pd.to_datetime('2022-{}-{}'.format(end.month, end.day))
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
    return df

def check_if_weather_data_exists():
    query = text("""SELECT EXISTS(SELECT 1 FROM people_sun.weatherdata LIMIT 1) as 'Exists';""")
    with get_sync_session_maker() as sync_db:
        res = sync_db.execute(query)
    results = res.scalars().all()
    ans = bool(results[0])
    return ans

