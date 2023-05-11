import pandas as pd
from sqlalchemy import delete, text
from fastapi_app.io.db import models
from fastapi_app.io.db.database import get_async_session_maker
from fastapi_app.io.db.queries import get_df
from sqlalchemy import update


async def merge_model(model):
    async with get_async_session_maker() as async_db:
        await async_db.merge(model)
        await async_db.commit()


async def update_model_by_user_id(model):
    stmt = (update(model.metadata.tables[model.__name__().lower()])
            .where(model.__mapper__.primary_key[0] == model.id).values(**model.to_dict()))
    async with get_async_session_maker() as async_db:
        await async_db.execute(stmt)
        await async_db.commit()


async def insert_links_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    model_class = models.Links
    await remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    await _insert_df(table='links', df=df, if_exists='update')


async def insert_nodes_df(df, user_id, project_id, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    model_class = models.Nodes
    if replace:
        await remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    await _insert_df('nodes', df, if_exists='update')


async def insert_energysystemdesign_df(df, user_id, project_id, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    model_class = models.EnergySystemDesign
    if replace:
        await remove(model_class, user_id, project_id)
    df['id'] = int(user_id)
    df['project_id'] = int(project_id)
    await _insert_df('energysystemdesign', df, if_exists='update')


async def insert_results_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.Results
        await remove(model_class, user_id, project_id)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        await _insert_df('results', df, if_exists='update')


async def insert_demand_coverage_df(df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        model_class = models.DemandCoverage
        await remove(model_class, user_id, project_id)
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        await _insert_df('demandcoverage', df, if_exists='update')


async def insert_df(model_class, df, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    df = df.dropna(how='all', axis=0)
    if not df.empty:
        await remove(model_class, user_id, project_id)
        if hasattr(model_class, 'dt') and 'dt' not in df.columns:
            df.index.name = 'dt'
            df = df.reset_index()
        df['id'] = int(user_id)
        df['project_id'] = int(project_id)
        await _insert_df(model_class.__name__.lower(), df, if_exists='update')


async def remove(model_class, user_id, project_id):
    user_id, project_id = int(user_id), int(project_id)
    query = delete(model_class).where(model_class.id == user_id, model_class.project_id == project_id)
    async with get_async_session_maker() as async_db:
        await async_db.execute(query)
        await async_db.commit()


async def remove_account(user_email, user_id):
    async with get_async_session_maker() as async_db:
        for model_class in [models.User,
                            models.ProjectSetup,
                            models.GridDesign,
                            models.EnergySystemDesign,
                            models.Nodes,
                            models.Links,
                            models.Results,
                            models.DemandCoverage,
                            models.EnergyFlow,
                            models.Emissions,
                            models.DurationCurve]:
            query = delete(model_class).where(model_class.id == user_id)
            await async_db.execute(query)
            await async_db.commit()


async def remove_project(user_id, project_id):
    for model_class in [models.Nodes, models.Links, models.Results, models.DemandCoverage, models.EnergyFlow,
                        models.Emissions, models.DurationCurve, models.ProjectSetup, models.EnergySystemDesign,
                        models.GridDesign]:
        await remove(model_class, user_id, project_id)


async def update_nodes_and_links(nodes: bool, links: bool, inlet: dict, user_id, project_id, add=True, replace=True):
    user_id, project_id = int(user_id), int(project_id)
    if nodes:
        nodes = inlet
        try:
            df = pd.DataFrame.from_dict(nodes).round(decimals=6)
        except ValueError:
            df = pd.DataFrame(nodes, index=[0]).round(decimals=6)
        if add and replace:
            df_existing = await get_df(models.Nodes, user_id, project_id)
            if not df_existing.empty:
                df_existing = df_existing[(df_existing["node_type"] != "pole") &
                                          (df_existing["node_type"] != "power-house")]
            df_total = df_existing.append(df).drop_duplicates(subset=["latitude", "longitude", "node_type"], inplace=False)
        else:
            df_total = df
        if not df.empty:
            df["node_type"] = df["node_type"].astype(str)
            if df["node_type"].str.contains("consumer").sum() > 0:
                df_links = await get_df(models.Links, user_id, project_id)
                if not df_links.empty:
                    df_links.drop(labels=df_links.index, axis=0, inplace=True)
                    await insert_links_df(df_links, user_id, project_id)
        if not df_total.empty:
            df_total.latitude = df_total.latitude.map(lambda x: "%.6f" % x)
            df_total.longitude = df_total.longitude.map(lambda x: "%.6f" % x)
            df_total.surface_area = df_total.surface_area.map(lambda x: "%.2f" % x)
            df_total.peak_demand = df_total.peak_demand.map(lambda x: "%.3f" % x)
            df_total.average_consumption = df_total.average_consumption.map(lambda x: "%.3f" % x)
            # finally adding the refined dataframe (if it is not empty) to the existing csv file
            if len(df_total.index) != 0:
                if 'parent' in df_total.columns:
                    df_total['parent'] = df_total['parent'].replace('unknown', None)
                await insert_nodes_df(df_total, user_id, project_id, replace=replace)
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
            await insert_links_df(df, user_id, project_id)


async def sql_str_2_db(sql):
    async_session = get_async_session_maker()
    try:
        await async_session.execute(text(sql))
        await async_session.commit()
        await async_session.close()
    except Exception as err:
        if len(sql) > 500:
            sql = "\n(...)\n".join((sql[0:min(400, int(len(sql) / 2))], sql[-100:]))
        print("\n Something went wrong while trying to write to the database.\n\n Your query was:\n{0}".format(sql))
        await async_session.rollback()  # Revert everything that has been written so far
        raise Exception(err)


async def _insert_df(table: str, df, if_exists='update', chunk_size=None):
    if df.empty:
        return
    max_rows = int(150000 / len(df.columns))
    if isinstance(df, pd.DataFrame) and chunk_size is None and len(df.index) < max_rows:
        sql = df_2_sql(table, df, if_exists)
        await sql_str_2_db(sql)
    else:
        if isinstance(df, pd.DataFrame):
            n_rows = len(df.index)
            chunk_size = chunk_size if isinstance(chunk_size, int) else max_rows
            df_list = []
            for first in range(0, n_rows, chunk_size):
                last = first + chunk_size if first + chunk_size < n_rows else n_rows
                df_list.append(df.iloc[first:last, :])
        elif isinstance(df, list):
            df_list = df.copy()
            for df in df_list:
                sql = df_2_sql(table, df, if_exists)
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
