import time
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi_app.db.config import db_host, db_name, db_user_name, PW, db_port
from fastapi_app.db.models import Base
from sqlalchemy.exc import SQLAlchemyError
from mysql.connector import DatabaseError, ProgrammingError, InterfaceError


SQLALCHEMY_DATABASE_URL = 'mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(db_user_name, PW, db_host, db_port, db_name)

for i in range(40):
    try:
        engine = create_engine(SQLALCHEMY_DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
    except (SQLAlchemyError, DatabaseError, ProgrammingError, InterfaceError) as e:
        time.sleep(5)
    else:
        break



def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()


def sql_str_2_db(sql, cnx=None):
    try:
        cnx.execute(sql)
        cnx.commit()
    except Exception as err:
        if len(sql) > 500:
            sql = "\n(...)\n".join((sql[0:min(400, int(len(sql) / 2))], sql[-100:]))
        print("\n Something went wrong while trying to write to the database.\n\n Your query was:\n{0}".format(sql))
        cnx.rollback()  # Revert everything that has been written so far
        raise Exception(err)


def _insert_df(table: str, df, cnx, if_exists='update', chunk_size=None):
    if df.empty:
        return
    max_rows = int(150000 / len(df.columns))
    if isinstance(df, pd.DataFrame) and chunk_size is None and len(df.index) < max_rows:
        sql = df_2_sql(table, df, if_exists)
        sql_str_2_db(sql, cnx)
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
                sql_str_2_db(sql, cnx)


def df_2_sql(table, df, if_exists):
    data = df.to_numpy()
    col_names = df.columns.to_numpy().tolist()
    col_names = ['`{}`'.format(col) if any(char.isdigit() for char in col) else col for col in col_names]
    del_char = ["'", "[", "]"]  # unwanted characters that occur during the column generation
    columns = ''.join(i for i in str(col_names) if
                      i not in del_char)  # now columns correspond to "col1,col2,col3" to database-column names
    values = str(data.tolist()).replace("[", "(") \
                 .replace("]", ")") \
                 .replace("nan", "NULL") \
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
        query2 = query[-10:].replace(";", "ON DUPLICATE KEY UPDATE {};".format(col_rules.strip(", ")))
        query = ''.join([query1, query2])
        return query