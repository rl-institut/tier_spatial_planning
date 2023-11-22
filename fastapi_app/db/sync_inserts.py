import pandas as pd
import numpy as np
import calendar
import os
from sqlalchemy import delete, text
from sqlalchemy.exc import OperationalError
import xarray as xr
import pvlib
from feedinlib import era5
from fastapi_app.db import models
from fastapi_app.db.db_con import get_sync_session_maker, sync_engine
from fastapi_app.db.sync_queries import get_df, get_model_instance
from fastapi_app.db.inserts import df_2_sql
from fastapi_app.db import config
from fastapi_app.tools.solar_potential import download_weather_data, prepare_weather_data
import subprocess




def merge_model(model):
    new_engine = False
    for i in range(config.DB_RETRY_COUNT):
        try:
            with get_sync_session_maker(sync_engine, new_engine) as session:
                session.merge(model)
                session.commit()
                return
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{DB_RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < DB_RETRY_COUNT - 1:  # Don't wait after the last try
                time.sleep(RETRY_DELAY)
            else:
                raise e
                print(f"Failed to merge and commit after {DB_RETRY_COUNT} retries")


def execute_stmt(stmt):
    stmt = text(stmt) if isinstance(stmt, str) else stmt
    new_engine = False
    for i in range(config.DB_RETRY_COUNT):
        try:
            with get_sync_session_maker(sync_engine, new_engine) as session:
                session.execute(stmt)
                session.commit()
                return
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{DB_RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < DB_RETRY_COUNT - 1:  # Don't wait after the last try
                time.sleep(RETRY_DELAY)
            else:
                raise e
                print(f"Failed to merge and commit after {DB_RETRY_COUNT} retries")


def update_nodes_and_links(nodes: bool, links: bool, inlet: dict, user_id, project_id, add=True, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    if nodes:
        nodes = inlet
        df = nodes.round(decimals=6)
        if add and replace:
            nodes_existing = get_model_instance(models.Nodes, user_id, project_id)
            if nodes_existing is not None:
                df_existing = pd.read_json(nodes_existing.data)
            else:
                df_existing = pd.DataFrame()
            if not df_existing.empty:
                df_existing = df_existing[(df_existing["node_type"] != "pole") &
                                          (df_existing["node_type"] != "power-house")]
            df_total = pd.concat([df, df_existing], ignore_index=True, axis=0)
            df_total = df_total.drop_duplicates(subset=["latitude", "longitude", "node_type"])
        else:
            df_total = df
        if not df.empty:
            df["node_type"] = df["node_type"].astype(str)
            if df["node_type"].str.contains("consumer").sum() > 0:
                df_links = get_df(models.Links, user_id, project_id)
                if not df_links.empty:
                    df_links.drop(labels=df_links.index, axis=0, inplace=True)
                    links = models.Links()
                    links.id = user_id
                    links.project_id = project_id
                    links.data = df_links.reset_index(drop=True).to_json()
                    merge_model(links)
        if not df_total.empty:
            df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
            df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
            # finally adding the refined dataframe (if it is not empty) to the existing csv file
            if len(df_total.index) != 0:
                if 'parent' in df_total.columns:
                    df_total['parent'] = df_total['parent'].where(df_total['parent'] != 'unknown', None)
                nodes = models.Nodes()
                nodes.id = user_id
                nodes.project_id = project_id
                nodes.data = df_total.reset_index(drop=True).to_json()
                merge_model(nodes)
    if links:
        links = inlet
        # defining the precision of data
        df = pd.DataFrame.from_dict(links)
        df.lat_from = df.lat_from.map(lambda x: "%.6f" % x)
        df.lon_from = df.lon_from.map(lambda x: "%.6f" % x)
        df.lat_to = df.lat_to.map(lambda x: "%.6f" % x)
        df.lon_to = df.lon_to.map(lambda x: "%.6f" % x)
        # adding the links to the existing csv file
        if len(df.index) != 0:
            links = models.Links()
            links.id = user_id
            links.project_id = project_id
            links.data = df.reset_index(drop=True).to_json()
            merge_model(links)


def _insert_df(table: str, df, if_exists='update', chunk_size=None):
    if df.empty:
        return
    max_rows = int(150000 / len(df.columns))
    if isinstance(df, pd.DataFrame) and chunk_size is None and len(df.index) < max_rows:
        sql = df_2_sql(table, df, if_exists)
        sql_str_2_db(sql)
    else:
        n_rows = len(df.index)
        chunk_size = chunk_size if isinstance(chunk_size, int) else max_rows
        for first in range(0, n_rows, chunk_size):
            last = first + chunk_size if first + chunk_size < n_rows else n_rows
            sql = df_2_sql(table, df.iloc[first:last, :], if_exists)
            sql_str_2_db(sql)


def sql_str_2_db(sql):
    stmt = text(sql)
    execute_stmt(stmt)


def remove(model_class, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    stmt = delete(model_class).where(model_class.id == user_id, model_class.project_id == project_id)
    execute_stmt(stmt)


def insert_results_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.Results
        remove(model_class, user_id, project_id)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        _insert_df('results', df, if_exists='update')


def insert_df(model_class, df, user_id=None, project_id=None, ts=True):
    if user_id is not None and project_id is not None:
        user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        if user_id is not None and project_id is not None:
            remove(model_class, user_id, project_id)
            df['id'] = int(user_id)
            df['project_id'] = int(project_id)
        if hasattr(model_class, 'dt') and 'dt' not in df.columns:
            df.index.name = 'dt'
            df = df.reset_index()
        _insert_df(model_class.__name__.lower(), df, if_exists='update')

def _from_netcdf4_file(file_name):
    if not os.path.exists(file_name):
        print(f"\nnetcdf4 file not found: {file_name}\n")
        return  False# Exit the function if file does not exist
    ds = xr.open_dataset(file_name, engine='netcdf4')
    df = era5.format_pvlib(ds)
    df = df.reset_index()
    df = df.rename(columns={'time': 'dt', 'latitude': 'lat', 'longitude': 'lon'})
    df = df.set_index(['dt'])
    def get_all_locations(ds):
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        lon_grid, lat_grid = np.meshgrid(lat, lon)
        grid_points = np.stack((lat_grid, lon_grid), axis=-1)
        grid_points = grid_points.reshape(-1, 2)
        return grid_points
    df['dni'] = np.nan
    grid_points = get_all_locations(ds)
    for lon, lat in grid_points:
        mask = (df['lat'] == lat) & (df['lon'] == lon)
        tmp_df = df.loc[mask]
        solar_position = pvlib.solarposition.get_solarposition(time=tmp_df.index,
                                                               latitude=lat,
                                                               longitude=lon)
        df.loc[mask, 'dni'] = pvlib.irradiance.dni(ghi=tmp_df['ghi'],
                                                   dhi=tmp_df['dhi'],
                                                   zenith=solar_position['apparent_zenith']).fillna(0)
    df = df.reset_index()
    df['dt'] = df['dt'] - pd.Timedelta('30min')
    df['dt'] = df['dt'].dt.tz_convert('UTC').dt.tz_localize(None)
    df.iloc[:, 3:] = df.iloc[:, 3:] + 0.0000001
    df.iloc[:, 3:] = df.iloc[:, 3:].round(1)
    df.loc[:, 'lon'] = df.loc[:, 'lon'].round(3)
    df.loc[:, 'lat'] = df.loc[:, 'lat'].round(7)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(str)
    insert_df(models.WeatherData, df)
    return True


def _db_sql_dump_import_weather_data():
    file_path = 'fastapi_app/data/weather/weatherdata.sql'
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                sql_commands = file.read().split(';')
            for command in sql_commands:
                if command.strip():
                    stmt = text(command)
                    execute_stmt(stmt)
            return True
    except Exception as e:
        print(f"Error executing SQL commands: {e}")
        return False


def dump_weather_data_into_db():
    if os.path.exists('fastapi_app/data/weather/weatherdata.sql'):
        print('\nDumping weather data into database. \n'
              'This usually takes a few minutes...\n'
              'Please wait and do not interrupt the process.\n')
        ans = _db_sql_dump_import_weather_data()
        if ans:
            return
    print('\nDownloading and dumping weather data into database. \n'
              'This usually takes a few minutes...\n'
              'Please wait and do not interrupt the process.\n')
    last_year = (pd.Timestamp.now() + pd.Timedelta(24 * 14, unit='H')).year - 1
    for month in range(1, 13, 3):  # Increment by 3
        print(f'Period starting {month} of 12 is being imported.')
        start_date = pd.Timestamp(year=last_year, month=month, day=1)
        end_month = month + 2  # Third month in the interval
        end_month = 12 if end_month > 12 else end_month  # Ensure it does not exceed December
        last_day_of_end_month = calendar.monthrange(last_year, end_month)[1]
        end_date = pd.Timestamp(year=last_year, month=end_month, day=last_day_of_end_month)
        try:
            data_xr = download_weather_data(start_date, end_date, country='Nigeria')
            df = prepare_weather_data(data_xr)
            insert_df(models.WeatherData, df)
            print(
                f"Data for period {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')} imported successfully")
        except subprocess.CalledProcessError as e:
            print("Error during import of weather data")
