import os
import time
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from fastapi_app.io.db.config import db_host, db_name, db_user_name, PW, db_port, DOMAIN
from fastapi_app.io.db.models import Base
from sqlalchemy.exc import SQLAlchemyError
from mysql.connector import DatabaseError, ProgrammingError, InterfaceError, OperationalError
from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError, ConfigurationError, ConnectionFailure

BASE_URL = 'mysql+package://{}:{}@{}:{}/{}'.format(db_user_name, PW, db_host, db_port, db_name)
SYNC_DB_URL = BASE_URL.replace('package', 'mysqlconnector')
ASYNC_DB_URL = BASE_URL.replace('package', 'aiomysql')



for i in range(400):
    try:
        sync_engine = create_engine(SYNC_DB_URL)
        Base.metadata.create_all(bind=sync_engine)
        async_engine = create_async_engine(ASYNC_DB_URL, pool_size=30, max_overflow=150, pool_timeout=30)
    except (SQLAlchemyError, DatabaseError, ProgrammingError, InterfaceError) as e:
        time.sleep(5)
    else:
        break

if bool(os.environ.get('DOCKERIZED')):
    for i in range(10):
        try:
            mongo_uri = 'mongodb://{}:{}@mongo:{}'.format(os.environ.get('MONGO_USER'),
                                                                          PW,
                                                                          os.environ.get('MONGO_PORT'))
            client = MongoClient(mongo_uri)
            db = client['admin']
            domains_collection = db['domains']
            document = {'title': DOMAIN,
                        'id': 'e6b4dbf9-7401-4172-8212-f6a13cd5f962',
                        'created': datetime.datetime.now(),
                        'updated': datetime.datetime.now()}
            query_filter = {"id": document["id"]}
            result = domains_collection.update_one(query_filter, {"$setOnInsert": document}, upsert=True)
        except (PyMongoError, ServerSelectionTimeoutError, ConfigurationError, ConnectionFailure) as e:
            time.sleep(5)
        else:
            break




def get_async_session_maker(async_engine, new_engine=False):
    if new_engine:
        async_engine = create_async_engine(ASYNC_DB_URL, pool_size=5, max_overflow=150, pool_timeout=30)
    async_sessionmaker = scoped_session(sessionmaker(bind=async_engine,
                                                     class_=AsyncSession))
    return async_sessionmaker()


def get_sync_session_maker(sync_engine, new_engine=False):
    if new_engine:
        sync_engine = create_engine(SYNC_DB_URL)
    sync_session = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
    return sync_session()

