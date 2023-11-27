import pandas as pd
import asyncio
import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy import delete, text
from fastapi_app.db import sa_tables
from fastapi_app.db.connections import get_async_session_maker, async_engine
from fastapi_app.db.async_queries import get_model_instance, get_user_by_username, get_projects_of_user
from sqlalchemy import update
from fastapi_app.config import DB_RETRY_COUNT, RETRY_DELAY


async def merge_model(model):
    new_engine = False
    for i in range(DB_RETRY_COUNT):
        try:
            async with get_async_session_maker(async_engine, new_engine) as async_db:
                await async_db.merge(model)
                await async_db.commit()
                return
        except OperationalError as e:
            if i == 0:
                new_engine = True
            elif i < DB_RETRY_COUNT - 1:  # Don't wait after the last try
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise e


async def execute_stmt(stmt):
    new_engine = False
    for i in range(DB_RETRY_COUNT):
        try:
            async with get_async_session_maker(async_engine, new_engine) as async_db:
                await async_db.execute(stmt)
                await async_db.commit()
                return
        except OperationalError as e:
            if i == 0:
                new_engine = True
            elif i < DB_RETRY_COUNT - 1:  # Don't wait after the last try
                await asyncio.sleep(RETRY_DELAY)
            else:
                raise e

async def update_model_by_user_id(model):
    stmt = (update(model.metadata.tables[model.__name__().lower()])
            .where(model.__mapper__.primary_key[0] == model.id).values(**model.to_dict()))
    await execute_stmt(stmt)


async def insert_energysystemdesign_df(df, user_id, project_id, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    model_class = sa_tables.EnergySystemDesign
    if replace:
        await remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    await _insert_df('energysystemdesign', df, if_exists='update')


async def insert_results_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = sa_tables.Results
        await remove(model_class, user_id, project_id)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        await _insert_df('results', df, if_exists='update')


async def insert_df(model_class, df, user_id=None, project_id=None):
    if user_id is not None and project_id is not None:
        user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        if user_id is not None and project_id is not None:
            await remove(model_class, user_id, project_id)
            df['id'] = int(user_id)
            df['project_id'] = int(project_id)
        if hasattr(model_class, 'dt') and 'dt' not in df.columns:
            df.index.name = 'dt'
            df = df.reset_index()
        await _insert_df(model_class.__name__.lower(), df, if_exists='update')


async def remove(model_class, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    stmt = delete(model_class).where(model_class.id == user_id, model_class.project_id == project_id)
    await execute_stmt(stmt)


async def remove_account(user_id):
    for model_class in [sa_tables.User,
                        sa_tables.ProjectSetup,
                        sa_tables.GridDesign,
                        sa_tables.EnergySystemDesign,
                        sa_tables.Nodes,
                        sa_tables.Links,
                        sa_tables.Results,
                        sa_tables.DemandCoverage,
                        sa_tables.Demand,
                        sa_tables.EnergyFlow,
                        sa_tables.Emissions,
                        sa_tables.DurationCurve]:
        stmt = delete(model_class).where(model_class.id == user_id)
        await execute_stmt(stmt)


async def remove_project(user_id, project_id):
    for model_class in [sa_tables.Nodes, sa_tables.Links, sa_tables.Results, sa_tables.DemandCoverage, sa_tables.EnergyFlow,
                        sa_tables.Emissions, sa_tables.DurationCurve, sa_tables.ProjectSetup, sa_tables.EnergySystemDesign,
                        sa_tables.GridDesign]:
        await remove(model_class, user_id, project_id)


async def sql_str_2_db(sql):
    stmt = text(sql)
    await execute_stmt(stmt)


async def _insert_df(table: str, df, if_exists='update', chunk_size=None):
    if df.empty:
        return
    max_rows = int(150000 / len(df.columns))
    if isinstance(df, pd.DataFrame) and chunk_size is None and len(df.index) < max_rows:
        sql = df_2_sql(table, df, if_exists)
        await sql_str_2_db(sql)
    else:
        n_rows = len(df.index)
        chunk_size = chunk_size if isinstance(chunk_size, int) else max_rows
        for first in range(0, n_rows, chunk_size):
            last = first + chunk_size if first + chunk_size < n_rows else n_rows
            sql = df_2_sql(table, df.iloc[first:last, :], if_exists)
            await sql_str_2_db(sql)


def df_2_sql(table, df, if_exists):
    if table == 'results':
        df = df.astype(float)
    data = df.to_numpy()
    col_names = df.columns.to_numpy().tolist()
    col_names = ['`{}`'.format(col) if any(char.isdigit() for char in col) else col for col in col_names]
    del_char = ["'", "[", "]"]  # unwanted characters that occur during the column generation
    columns = ''.join(i for i in str(col_names) if
                      i not in del_char)  # now columns correspond to "col1,col2,col3" to database-column names
    values = str(data.tolist()).replace("[", "(") \
                 .replace("]", ")") \
                 .replace("nan", "NULL") \
                 .replace('n.a.', "NULL") \
                 .replace("None", "NULL") \
                 .replace("NaT", "NULL") \
                 .replace("<NA>", "NULL") \
                 .replace("\'NULL\'", "NULL")[1:-1]
    sql = "INSERT INTO {0}({1}) VALUES {2};".format(table, columns, values)
    if if_exists is not None:
        sql = handle_duplicates(if_exists, sql, col_names)
    return sql


def handle_duplicates(if_exists, query, col_names):
    """ Calling the method 'write' without specifying the argument 'if_exists' (or if_exists=None) will raise
        an error if a primary_key already exists. If the arguments value is 'update', rows with duplicate
        keys will be overwritten. """
    if if_exists not in ["update", None]:
        raise ValueError("\"{}\" is no allowed value for argument \"if_exists\" of "
                         "function \"write\"! Allowed values are: {}"
                         .format(if_exists, "update, None"))
    if if_exists == "update":
        col_rules = "".join('{} = VALUES({}), '.format(col_name, col_name) for col_name in col_names)
        query1 = query[:-10]
        query2 = query[-10:].replace(";", "AS alias ON DUPLICATE KEY UPDATE {};".format(col_rules.strip(", ")
                                                                                        .replace('VALUES(', 'alias.')
                                                                                        .replace(')', '')))
        query = ''.join([query1, query2])
        return query


async def insert_example_project(user_id):
    example = await get_user_by_username('default_example')
    projects = await get_projects_of_user(user_id)
    if example is not None and hasattr(example, 'id') and len(projects) == 0:
        await _copy_project(example.id, user_id, 0, 0)


async def copy_project(user_id, project_id):
    projects = await get_projects_of_user(user_id)
    next_project_id = max(project.project_id for project in projects) + 1 if len(projects) > 0 else 0
    await _copy_project(user_id, user_id, project_id, next_project_id)


async def _copy_project(user_from_id, user_to_id, project_from_id, project_to_id):
    for model_class in [sa_tables.Nodes, sa_tables.Links, sa_tables.Results, sa_tables.DemandCoverage, sa_tables.EnergyFlow,
                        sa_tables.Emissions, sa_tables.DurationCurve, sa_tables.ProjectSetup, sa_tables.EnergySystemDesign,
                        sa_tables.GridDesign, sa_tables.Demand]:
        model_instance = await get_model_instance(model_class, user_from_id, project_from_id, 'all')
        if model_class == sa_tables.ProjectSetup:
            time_now = datetime.datetime.now()
            time_now \
                = datetime.datetime(time_now.year, time_now.month, time_now.day, time_now.hour, time_now.minute)
            model_instance[0].created_at = time_now
            model_instance[0].updated_at = time_now
            model_instance[0].project_name = 'Copy of {}'.format(model_instance[0].project_name)
        for e in model_instance:
            data = {key: value for key, value in e.__dict__.items() if not key.startswith('_')}
            new_e = model_class(**data)
            new_e.id = user_to_id
            new_e.project_id = project_to_id
            await merge_model(new_e)
