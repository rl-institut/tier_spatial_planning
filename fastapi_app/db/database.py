from typing import Any
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi_app.db.config import host, db_name, user_name, password, port


# nodes database
SQLALCHEMY_DATABASE_URL = 'mysql+mysqlconnector://{}:{}@{}:{}/{}'\
    .format(user_name, password, host, port, db_name)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@as_declarative()
class Base:
    id: Any
    __name__: str

    # generate tablename from classname
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()


def get_db():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
