import calendar
import os
import time
import warnings

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.exc import OperationalError

from fastapi_app.config import DB_RETRY_COUNT, RETRY_DELAY
from fastapi_app.db import sa_tables
from fastapi_app.db.async_inserts import df_2_sql
from fastapi_app.db.connections import get_sync_session_maker, sync_engine
from fastapi_app.inputs.solar_potential import download_weather_data, prepare_weather_data


def merge_model(model):
    new_engine = False
    for i in range(DB_RETRY_COUNT):
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
                print(f"Failed to merge and commit after {DB_RETRY_COUNT} retries")
                raise e


def execute_stmt(stmt):
    stmt = text(stmt) if isinstance(stmt, str) else stmt
    new_engine = False
    for i in range(DB_RETRY_COUNT):
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
                print(f"Failed to merge and commit after {DB_RETRY_COUNT} retries")
                raise e


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


def insert_df(model_class, df, user_id=None, project_id=None):
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


def db_sql_dump_import(file_path):
    print('\nImporting sql dump file \n')
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                sql_commands = []
                command = ""
                for line in file:
                    if line.startswith('--') or not line.strip():
                        continue
                    command += line
                    if ';' in line:
                        sql_commands.append(command)
                        command = ""
            for cmd in sql_commands:
                if cmd.strip():
                    stmt = text(cmd)
                    execute_stmt(stmt)  # Execute each statement
            print("\nAll SQL commands executed successfully.\n")
            return True
    except Exception as e:
        print(f"Error executing SQL commands: {e}")
        return False


def update_weather_db(country='Nigeria', year=None):
    if year is not None and year >= (pd.Timestamp.now() + pd.Timedelta(24 * 7, unit='H')).year:
        raise Exception("This function excepts available weather data for a entire year, "
                        "but for {} that data is not yet available".format(year))
    elif year != 2022:
        warnings.warn("Currently, only simulation the year 2022 is possible. Refer to the comments for "
                      "detailed explanations.")
    year = (pd.Timestamp.now() + pd.Timedelta(24 * 14, unit='H')).year - 1 if year is None else int(year)
    year = 2022 # so fast demand data is only available for 2022 and start_date is always 2022-01-01 and max. duration
    # is one year (see func 'save_project_setup' in static/js/backend_communications.js )
    for month in range(1, 13, 3):  # Increment by 3
        start_date = pd.Timestamp(year=year, month=month, day=1) - pd.Timedelta(25, unit='H')
        end_month = month + 2  # Third month in the interval
        end_month = 12 if end_month > 12 else end_month  # Ensure it does not exceed December
        last_day_of_end_month = calendar.monthrange(year, end_month)[1]
        end_date = pd.Timestamp(year=year, month=end_month, day=last_day_of_end_month) + pd.Timedelta(25, unit='H')
        file_name = 'cfd_weather_data_{}.nc'.format(start_date.strftime('%Y-%m'))
        data_xr = download_weather_data(start_date, end_date, country=country, target_file=file_name).copy()
        df = prepare_weather_data(data_xr)
        insert_df(sa_tables.WeatherData, df)
