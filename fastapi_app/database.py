from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi_app.config import host, db_name, user_name, password, port


# nodes database
SQLALCHEMY_DATABASE_URL = 'mysql+mysqlconnector://{}:{}@{}host:{}/{}'\
    .format(user_name, password, host, port, db_name)

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
