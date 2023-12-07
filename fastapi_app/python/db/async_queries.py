import asyncio
import decimal

import pandas as pd
import sqlalchemy as sa
from jose import jwt, JWTError
from sqlalchemy import select
from sqlalchemy.exc import OperationalError, NoResultFound

from fastapi_app.python import config
from fastapi_app.python.config import DB_RETRY_COUNT, RETRY_DELAY
from fastapi_app.python.db import sa_tables
from fastapi_app.python.db.connections import get_async_session_maker, async_engine

"""
Asynchronous functions performing database queries. When database operations are triggered by FastAPI routes, 
they are executed as asynchronous functions to avoid blocking the execution of other Python code. In contrast, when 
database functions are executed by energy system models (GridOptimizer, EnergySystemOptimizer) processed within a 
Docker network by the Celery task queue, synchronous database operations are used (see modules sync_inserts, 
sync_queries)
"""

async def get_user_by_username(username):
    query = select(sa_tables.User).where(sa_tables.User.email == username)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_user_by_id(user_id):
    query = select(sa_tables.User).where(sa_tables.User.id == user_id)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_user_by_guid(guid):
    query = select(sa_tables.User).where(sa_tables.User.guid == guid)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_project_name_by_id(user_id, project_id):
    query = select(sa_tables.ProjectSetup).where(sa_tables.ProjectSetup.id == user_id,
                                                 sa_tables.ProjectSetup.project_id == project_id)
    project = await _execute_with_retry(query, which='first')
    return project


async def get_max_project_id_of_user(user_id):
    subqry = select(sa.func.max(sa_tables.ProjectSetup.project_id)).filter(
        sa_tables.ProjectSetup.id == user_id).as_scalar()
    query = select(sa_tables.ProjectSetup).filter(sa_tables.ProjectSetup.id == user_id,
                                                  sa_tables.ProjectSetup.project_id == subqry)
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


async def get_projects_of_user(user_id):
    user_id = int(user_id)
    query = select(sa_tables.ProjectSetup).where(sa_tables.ProjectSetup.id == user_id)
    projects = await _execute_with_retry(query, which='all')
    return projects


async def get_model_instance(model, user_id, project_id, which='first'):
    user_id, project_id = int(user_id), int(project_id)
    query = select(model).where(model.id == user_id, model.project_id == project_id)
    model_instance = await _execute_with_retry(query, which=which)
    return model_instance


async def get_input_df(user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    project_setup = await get_df(sa_tables.ProjectSetup, user_id, project_id, is_timeseries=False)
    grid_design = await get_df(sa_tables.GridDesign, user_id, project_id, is_timeseries=False)
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


async def get_user_from_token(token):
    try:
        payload = jwt.decode(token, config.KEY_FOR_ACCESS_TOKEN, algorithms=[config.TOKEN_ALG])
        username = payload.get("sub")
    except JWTError:
        return None
    # if isinstance(db, scoped_session):
    query = select(sa_tables.User).where(sa_tables.User.email == username)
    user = await _execute_with_retry(query, which='first')
    return user


async def get_user_from_task_id(task_id):
    query = select(sa_tables.User).where(sa_tables.User.task_id == task_id)
    user = await _execute_with_retry(query, which='first')
    return user


async def _execute_with_retry(query, which='first'):
    new_engine = False
    for i in range(DB_RETRY_COUNT):
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
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{DB_RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < DB_RETRY_COUNT - 1:  # Don't wait after the last try
                await asyncio.sleep(RETRY_DELAY)
            else:
                print(f"Failed to execute query after {DB_RETRY_COUNT} retries")
                return None
        except NoResultFound:
            return None


async def check_data_availability(user_id, project_id):
    project_setup = await get_model_instance(sa_tables.ProjectSetup, user_id, project_id)
    if project_setup is None:
        return False, '/project_setup/?project_id=' + str(project_id)
    nodes = await get_model_instance(sa_tables.Nodes, user_id, project_id)
    nodes_df = pd.read_json(nodes.data) if nodes is not None else None
    if nodes_df is None or nodes_df.empty or nodes_df[nodes_df['node_type'] == 'consumer'].index.__len__() == 0:
        return False, '/consumer_selection/?project_id=' + str(project_id)
    demand_opt_dict = await get_model_instance(sa_tables.Demand, user_id, project_id)
    if demand_opt_dict is None or pd.isna(demand_opt_dict.household_option):
        return False, '/demand_estimation/?project_id=' + str(project_id)
    grid_design = await get_model_instance(sa_tables.GridDesign, user_id, project_id)
    if grid_design is None or pd.isna(grid_design.pole_lifetime):
        return False, '/grid_design/?project_id=' + str(project_id)
    energy_system_design = await get_model_instance(sa_tables.EnergySystemDesign, user_id, project_id)
    if energy_system_design is None or pd.isna(energy_system_design.battery__parameters__c_rate_in):
        return False, '/energy_system_design/?project_id=' + str(project_id)
    else:
        return True, None


async def pause_until_results_are_available(user_id, project_id, status):
    n_iter = 4 if status == 'unknown' else 2
    for i in range(n_iter):
        results = await get_model_instance(sa_tables.Results, user_id, project_id)
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
