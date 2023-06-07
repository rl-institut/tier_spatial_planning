import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.exc import OperationalError
from fastapi_app.io.db import models
from fastapi_app.io.db.database import get_sync_session_maker, sync_engine
from fastapi_app.io.db.sync_queries import get_df
from fastapi_app.io.db.inserts import df_2_sql
from fastapi_app.io.db.config import RETRY_COUNT, RETRY_DELAY



def merge_model(model):
    new_engine = False
    for i in range(RETRY_COUNT):
        try:
            with get_sync_session_maker(sync_engine, new_engine) as session:
                session.merge(model)
                session.commit()
                return
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < RETRY_COUNT - 1:  # Don't wait after the last try
                time.sleep(RETRY_DELAY)
            else:
                raise e
                print(f"Failed to merge and commit after {RETRY_COUNT} retries")


def execute_stmt(stmt):
    new_engine = False
    for i in range(RETRY_COUNT):
        try:
            with get_sync_session_maker(sync_engine, new_engine) as session:
                session.execute(stmt)
                session.commit()
                return
        except OperationalError as e:
            print(f'OperationalError occurred: {str(e)}. Retrying {i + 1}/{RETRY_COUNT}')
            if i == 0:
                new_engine = True
            elif i < RETRY_COUNT - 1:  # Don't wait after the last try
                time.sleep(RETRY_DELAY)
            else:
                raise e
                print(f"Failed to merge and commit after {RETRY_COUNT} retries")


def update_nodes_and_links(nodes: bool, links: bool, inlet: dict, user_id, project_id, add=True, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    if nodes:
        nodes = inlet
        try:
            df = pd.DataFrame.from_dict(nodes).round(decimals=6)
        except ValueError:
            df = pd.DataFrame(nodes, index=[0]).round(decimals=6)
        if add and replace:
            df_existing = get_df(models.Nodes, user_id, project_id)
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
                    insert_links_df(df_links, user_id, project_id)
        if not df_total.empty:
            df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
            df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
            df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
            df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
            df_total.average_consumption = df_total.average_consumption.map(lambda x: "%.3f" % x)
            # finally adding the refined dataframe (if it is not empty) to the existing csv file
            if len(df_total.index) != 0:
                if 'parent' in df_total.columns:
                    df_total['parent'] = df_total['parent'].where(df_total['parent'] != 'unknown', None)
                insert_nodes_df(df_total, user_id, project_id, replace=replace)
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
            insert_links_df(df, user_id, project_id)

def insert_links_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    model_class = models.Links
    remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    _insert_df(table='links', df=df, if_exists='update')


def insert_nodes_df(df, user_id, project_id, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    model_class = models.Nodes
    if replace:
        remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    _insert_df('nodes', df, if_exists='update')


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


def insert_demand_coverage_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.DemandCoverage
        remove(model_class, user_id, project_id)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        _insert_df('demandcoverage', df, if_exists='update')