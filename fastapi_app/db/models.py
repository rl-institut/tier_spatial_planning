from sqlalchemy import Boolean, Column, Integer, String, Numeric
# from sqlalchemy.orm import relationship
from pydantic import BaseModel
from fastapi_app.db.database import Base
from typing import List, Dict, Union


class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
