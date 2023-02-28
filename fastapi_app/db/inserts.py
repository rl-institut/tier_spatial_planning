import os
import pandas as pd
from fastapi_app.db import models
from fastapi_app.db.queries import get_nodes_df, get_links_df
from fastapi_app.db.database import _insert_df
from sqlalchemy import select, delete


async def insert_links_df(df, user_id, project_id, db):
    model_class = models.Links
    await remove(model_class, user_id, project_id, db)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    _insert_df(table='links', df=df, cnx=db, if_exists='update')


async def insert_nodes_df(df, user_id, project_id, db, replace=True):
    model_class = models.Nodes
    if replace:
        await remove(model_class, user_id, project_id, db)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    _insert_df('nodes', df, db, if_exists='update')


def insert_results_df(df, user_id, project_id, db):
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.Results
        remove(model_class, user_id, project_id, db)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        _insert_df('results', df, db, if_exists='update')


def insert_demand_coverage_df(df, user_id, project_id, db):
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.DemandCoverage
        remove(model_class, user_id, project_id, db)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        _insert_df('demandcoverage', df, db, if_exists='update')


def insert_df(model_class, df, user_id, project_id, db):
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        remove(model_class, user_id, project_id, db)
        if hasattr(model_class, 'dt') and 'dt' not in df.columns:
            df.index.name = 'dt'
            df = df.reset_index()
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        _insert_df(model_class.__name__.lower(), df, db, if_exists='update')


async def remove(model_class, user_id, project_id, db):
    if model_class in [models.Nodes, models.Links, models.DemandCoverage]:
        query = delete(model_class).where(model_class.id == user_id, model_class.project_id == project_id)
        async with db() as async_db:
            await async_db.execute(query)
            await async_db.commit()


async def update_nodes_and_links(nodes: bool, links: bool, inlet: dict, user_id, project_id, db, add=True, replace=True):
    if nodes:
        nodes = inlet
        df = pd.DataFrame.from_dict(nodes).round(decimals=6)
        if add and replace:
            df_existing = await get_nodes_df(user_id, project_id, db)
            if not df_existing.empty:
                df_existing = df_existing[(df_existing["node_type"] != "pole") &
                                          (df_existing["node_type"] != "power-house")]
            df_total = df_existing.append(df).drop_duplicates(subset=["latitude", "longitude", "node_type"], inplace=False)
        else:
            df_total = df
        if df["node_type"].str.contains("consumer").sum() > 0:
            df_links = await get_links_df(user_id, project_id, db)
            if not df_links.empty:
                df_links.drop(labels=df_links.index, axis=0, inplace=True)
                await insert_links_df(df_links, user_id, project_id, db)
        df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
        df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
        df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
        df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
        df_total.average_consumption = df_total.average_consumption.map(lambda x: "%.3f" % x)
        # finally adding the refined dataframe (if it is not empty) to the existing csv file
        if len(df_total.index) != 0:
            if 'parent' in df_total.columns:
                df_total['parent'] = df_total['parent'].replace('unknown', None)
            await insert_nodes_df(df_total, user_id, project_id, db, replace=replace)
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
            await insert_links_df(df, user_id, project_id, db)
