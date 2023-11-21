import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from fastapi_app.db.models import Base
from sqlalchemy.exc import SQLAlchemyError
from mysql.connector import DatabaseError, ProgrammingError, InterfaceError
from fastapi_app.db.config import DB_USER_NAME, PW, DB_HOST, DB_PORT, DB_NAME, DOMAIN


BASE_URL = 'mysql+package://{}:{}@{}:{}/{}'.format(DB_USER_NAME, PW, DB_HOST, DB_PORT, DB_NAME)
SYNC_DB_URL = BASE_URL.replace('package', 'mysqlconnector')
ASYNC_DB_URL = BASE_URL.replace('package', 'aiomysql')


for i in range(100):
    try:
        sync_engine = create_engine(SYNC_DB_URL)
        Base.metadata.create_all(bind=sync_engine)
        async_engine = create_async_engine(ASYNC_DB_URL, pool_size=30, max_overflow=150, pool_timeout=30)
    except (SQLAlchemyError, DatabaseError, ProgrammingError, InterfaceError) as e:
        print(e)
        print('Retry in 5 seconds...')
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
