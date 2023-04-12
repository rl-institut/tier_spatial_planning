import time
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from fastapi_app.db.config import db_host, db_name, db_user_name, PW, db_port
from fastapi_app.db.models import Base
from sqlalchemy.exc import SQLAlchemyError
from mysql.connector import DatabaseError, ProgrammingError, InterfaceError
from pymongo import MongoClient
from pymongo.errors import PyMongoError


BASE_URL = 'mysql+package://{}:{}@{}:{}/{}'.format(db_user_name, PW, db_host, db_port, db_name)
SYNC_DB_URL = BASE_URL.replace('package', 'mysqlconnector')
ASYNC_DB_URL = BASE_URL.replace('package', 'aiomysql')


for i in range(400):
    try:
        sync_engine = create_engine(SYNC_DB_URL)
        sync_session = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
        Base.metadata.create_all(bind=sync_engine)
        async_engine = create_async_engine(ASYNC_DB_URL, pool_size=20, )
        async_sessionmaker = scoped_session(sessionmaker(bind=async_engine,
                                                         class_=AsyncSession))
    except (SQLAlchemyError, DatabaseError, ProgrammingError, InterfaceError) as e:
        time.sleep(5)
    else:
        break


for i in range(4):
    try:
        client = MongoClient('mongodb://localhost:27017')
        db = client['ackee']
        domains_collection = db['domains']
        document = {'title': 'peoplesun.energietechnik.tu-berlin.de',
                  'id': 'e6b4dbf9-7401-4172-8212-f6a13cd5f962',
                  'created': datetime.datetime.now(),
                  'updated': datetime.datetime.now()}
        query_filter = {"id": document["id"]}
        result = domains_collection.update_one(query_filter, {"$setOnInsert": document}, upsert=True)
    except PyMongoError as e:
        time.sleep(5)
    else:
        break


def get_async_session_maker():
    async_engine = create_async_engine(ASYNC_DB_URL, pool_size=10, )
    async_sessionmaker = scoped_session(sessionmaker(bind=async_engine,
                                                     class_=AsyncSession))
    return async_sessionmaker()

def get_sync_session_maker():
    sync_engine = create_engine(SYNC_DB_URL)
    sync_session = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
    return sync_session()


def get_async_db():
    db = async_sessionmaker
    yield db
