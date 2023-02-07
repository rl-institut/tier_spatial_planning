import os
import pandas as pd
from fastapi_app.db import models
from fastapi_app.db.queries import get_nodes_df, get_links_df
from fastapi_app.db.database import insert_df


def insert_links_df(df, user_id, project_id, db):
    model_class = models.Links
    remove(model_class, user_id, project_id, db)
    df['id'] = user_id
    df['project_id'] = project_id
    insert_df('links', df, db, if_exists='update')


def insert_nodes_df(df, user_id, project_id, db, replace=True):
    model_class = models.Nodes
    if replace:
        remove(model_class, user_id, project_id, db)
    df['id'] = user_id
    df['project_id'] = project_id
    insert_df('nodes', df, db, if_exists='update')


def remove(model_class, user_id, project_id, db):
    if model_class == models.Nodes or model_class == models.Links:
        try:
            query = db.query(model_class).filter(model_class.id == user_id, model_class.project_id == project_id)
            query.delete()
        except Exception as e:
            db.rollback()
            raise e
        else:
            db.commit()


def update_nodes_and_links(nodes: bool, links: bool, inlet: dict, user_id, project_id, db, add=True, replace=True):
    if nodes:
        nodes = inlet
        df = pd.DataFrame.from_dict(nodes).round(decimals=6)
        if add and replace:
            df_existing = get_nodes_df(user_id, project_id, db)
            if not df_existing.empty:
                df_existing = df_existing[(df_existing["node_type"] != "pole") &
                                          (df_existing["node_type"] != "power-house")]
            df_total = df_existing.append(df).drop_duplicates(subset=["latitude", "longitude", "node_type"], inplace=False)
        else:
            df_total = df
        if df["node_type"].str.contains("consumer").sum() > 0:
            df_links = get_links_df(user_id, project_id, db)
            if not df_links.empty:
                df_links.drop(labels=df_links.index, axis=0, inplace=True)
                insert_links_df(df_links, user_id, project_id, db)
        df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
        df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
        df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
        df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
        df_total.average_consumption = df_total.average_consumption.map(lambda x: "%.3f" % x)
        # finally adding the refined dataframe (if it is not empty) to the existing csv file
        if len(df_total.index) != 0:
            if 'parent' in df_total.columns:
                df_total['parent'] = df_total['parent'].replace('unknown', None)
            insert_nodes_df(df_total, user_id, project_id, db, replace=replace)
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
            insert_links_df(df, user_id, project_id, db)
