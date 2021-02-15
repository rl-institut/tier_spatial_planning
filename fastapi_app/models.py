from sqlalchemy import Boolean, Column, Integer, String, Numeric
from sqlalchemy.orm import relationship

from database import Base


class Nodes(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)
    
    latitude = Column(Numeric(10, 4))
    longitude = Column(Numeric(10, 4))
    node_type = Column(String)
    fixed_type = Column(Boolean)
