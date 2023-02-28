import json
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import select
from fastapi_app.db import models
from fastapi_app.db.database import get_async_session_maker


def get_max_project_id_of_user(user_id, db):
    subqry = db.query(sa.func.max(models.ProjectSetup.project_id)).filter(models.ProjectSetup.id == user_id)
    qry = db.query(models.ProjectSetup).filter(models.ProjectSetup.id == user_id,
                                               models.ProjectSetup.project_id == subqry)
    res = qry.first()
    max_project_id = res.project_id if hasattr(res, 'project_id') else None
    return max_project_id


def next_project_id_of_user(user_id, db):
    max_project_id = get_max_project_id_of_user(user_id, db)
    if pd.isna(max_project_id):
        next_project_id = 0
    else:
        next_project_id = max_project_id + 1
    return next_project_id


async def get_project_of_user(user_id, db):
    async with db() as async_db:
        res = await async_db.execute(select(models.ProjectSetup).where(models.ProjectSetup.id == user_id))
        projects = res.scalars().all()
    return projects


async def get_project_setup_of_user(user_id, project_id, db):
    query = select(models.ProjectSetup).where(models.ProjectSetup.id == user_id,
                                                         models.ProjectSetup.project_id == project_id)
    async with db() as async_db:
        res = await async_db.execute(query)
        project_setup = res.scalars().first()
    return project_setup


async def get_nodes_df(user_id, project_id, db=None):
    query = select(models.Nodes).where(models.Nodes.id == user_id, models.Nodes.project_id == project_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
        nodes = res.scalars().fetchall()
    nodes = [node.get_dict() for node in nodes]
    df = pd.DataFrame.from_records(nodes).drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    return df


async def get_nodes_json(user_id, project_id):
    df = await get_nodes_df(user_id, project_id)
    nodes_json = json.loads(df.to_json())
    return nodes_json


async def get_links_df(user_id, project_id, db=None):
    query = select(models.Links).where(models.Links.id == user_id, models.Links.project_id == project_id)
    async with get_async_session_maker() as async_db:
        res = await async_db.execute(query)
    links = res.scalars().all()
    links = [link.get_dict() for link in links]
    df = pd.DataFrame.from_records(links).drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    return df


async def get_links_json(user_id, project_id):
    df = await get_links_df(user_id, project_id)
    nodes_json = json.loads(df.to_json())
    return nodes_json


async def get_grid_design_of_user(user_id, project_id, db):
    query = select(models.GridDesign).where(models.GridDesign.id == user_id,
                                                     models.GridDesign.project_id == project_id)
    async with db() as async_db:
        res = await async_db.execute(query)
        grid_design = res.scalars().first()
    return grid_design


async def get_input_df(user_id, project_id, db):
    project_setup = await get_project_setup_of_user(user_id, project_id, db)
    grid_design = await get_grid_design_of_user(user_id, project_id, db)
    df = pd.concat([project_setup.get_df(), grid_design.get_df()], axis=1).drop(columns=['id', 'project_id'])
    return df


def get_results_df(user_id, project_id, db):
    query = db.query(models.Results).filter(models.Results.id == user_id, models.Results.project_id == project_id)
    df = pd.read_sql(query.statement, db.bind).drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    return df


def get_demand_coverage_df(user_id, project_id, db):
    query = db.query(models.DemandCoverage).filter(models.DemandCoverage.id == user_id, models.DemandCoverage.project_id == project_id)
    df = pd.read_sql(query.statement, db.bind).drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    df = df.set_index('dt')
    return df


def get_df(model, user_id, project_id, db):
    query = db.query(model).filter(model.id == user_id, model.project_id == project_id)
    df = pd.read_sql(query.statement, db.bind).drop(columns=['id', 'project_id']).dropna(how='all', axis=0)
    if 'dt' in df.columns:
        df = df.set_index('dt')
    return df

def _read_sql(con, stmt):
    return pd.read_sql_query(stmt, con)