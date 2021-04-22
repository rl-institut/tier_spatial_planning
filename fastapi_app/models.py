from sqlalchemy import Boolean, Column, Integer, String, Numeric
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Optional
from database import Base

# Models


class Nodes(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)

    latitude = Column(Numeric(10, 4))
    longitude = Column(Numeric(10, 4))
    node_type = Column(String)
    fixed_type = Column(Boolean)


class Links(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    lat_from = Column(Numeric(10, 4))
    long_from = Column(Numeric(10, 4))
    lat_to = Column(Numeric(10, 4))
    long_to = Column(Numeric(10, 4))
    cable_type = Column(String)
    distance = Column(Numeric(10, 4))


class AddNodeRequest(BaseModel):
    latitude: float
    longitude: float
    node_type: str
    fixed_type: bool


class OptimizeGridRequest(BaseModel):
    price_meterhub: float
    price_household: float
    price_interhub_cable: float
    price_distribution_cable: float


class ShsIdentificationRequest(BaseModel):
    cable_price_per_meter_for_shs_mst_identification: float
    additional_connection_price_for_shs_mst_identification: float


class ValidateBoundariesRequest(BaseModel):
    boundary_coordinates: list
