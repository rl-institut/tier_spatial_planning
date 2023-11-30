import decimal
import time

import flatten_dict
import pandas as pd
import sqlalchemy.exc
from flatten_dict.splitters import make_splitter
from sqlalchemy import select, and_
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

from fastapi_app import config
from fastapi_app.python.db import sa_tables
from fastapi_app.python.db.connections import get_sync_session_maker, sync_engine


def get_project_setup_of_user(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(sa_tables.ProjectSetup).where(and_(sa_tables.ProjectSetup.id == user_id,
                                                      sa_tables.ProjectSetup.project_id == project_id))
    project_setup = _execute_with_retry(query, which='first')
    return project_setup


def get_input_df(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    project_setup = get_df(sa_tables.ProjectSetup, user_id, project_id, is_timeseries=False)
    grid_design = get_df(sa_tables.GridDesign, user_id, project_id, is_timeseries=False)
    df = pd.concat([project_setup, grid_design], axis=1)
    return df


def get_df(model, user_id, project_id, is_timeseries=True):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    df = _get_df(query, is_timeseries=is_timeseries)
    return df


def _get_df(query, is_timeseries=True):
    if is_timeseries:
        results = _execute_with_retry(query, which='all')
        results = [result.to_dict() for result in results]
    else:
        results = _execute_with_retry(query, which='one')
        results = [results.to_dict()]
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


def get_model_instance(model, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    model_instance = _execute_with_retry(query, which='first')
    return model_instance


def get_user_by_id(user_id):
    query = select(sa_tables.User).where(sa_tables.User.id == user_id)
    user = _execute_with_retry(query, which='first')
    return user

def get_user_by_username(username):
    query = select(sa_tables.User).where(sa_tables.User.email == username)
    user = _execute_with_retry(query, which='first')
    return user


def get_weather_data(lat, lon, start, end):
    index = pd.date_range(start, end, freq='1H')
    ts_changed = False
    if end > pd.to_datetime('2023-03-01'):
        end = pd.to_datetime('2022-{}-{}'.format(start.month, start.day)) + (end - start)
        start = pd.to_datetime('2022-{}-{}'.format(start.month, start.day))
        ts_changed = True
    model = sa_tables.WeatherData
    closest_lat, closest_lon = get_closest_grid_point(lat, lon)
    query = select(model).where(model.lat == closest_lat, model.lon == closest_lon, model.dt >= start, model.dt <= end)
    df = _get_df(query, is_timeseries=True)
    if ts_changed:
        df.index = index
    return df


def get_energy_system_design(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = select(sa_tables.EnergySystemDesign).where(and_(sa_tables.EnergySystemDesign.id == user_id,
                                                            sa_tables.EnergySystemDesign.project_id == project_id))
    model_inst = _execute_with_retry(query, which='first')
    df = model_inst.to_df()
    if df.empty:
        df = sa_tables.Nodes().to_df().iloc[0:0]
    df = df.drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    energy_system_design = flatten_dict.unflatten(df.to_dict('records')[0], splitter=make_splitter('__'))
    return energy_system_design


def check_if_weather_data_exists():
    check_table_query = text(f"""
            SELECT COUNT(*)
            FROM information_schema.tables 
            WHERE table_schema = '{config.DB_NAME}' 
            AND table_name = 'weatherdata';
        """)
    res = bool(_execute_with_retry(check_table_query, which='first'))
    if res is False:
        return False
    query = text("""SELECT COUNT(*) FROM {}.weatherdata;""".format(config.DB_NAME))
    res = _execute_with_retry(query, which='first')
    if isinstance(res, list):
        res = res[0]
    if isinstance(res, int):
        row_count = res
    else:
        row_count = 0
    return row_count >= 28151136


def _execute_with_retry(query, which='first'):
    new_engine = False
    for i in range(config.DB_RETRY_COUNT):
        try:
            with get_sync_session_maker(sync_engine, new_engine) as session:
                res = session.execute(query)
                if which == 'first':
                    res = res.scalars().first()
                elif which == 'all':
                    res = res.scalars().all()
                elif which == 'one':
                    try:
                        res = res.scalars().one()
                    except sqlalchemy.exc.NoResultFound:
                        return None
                return res
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{config.DB_RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < config.DB_RETRY_COUNT - 1:  # Don't wait after the last try
                time.sleep(config.RETRY_DELAY)
            else:
                print(f"Failed to execute query after {config.DB_RETRY_COUNT} retries")
                raise e


def get_closest_grid_point(lat, lon):
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
    return closest_lat, closest_lon