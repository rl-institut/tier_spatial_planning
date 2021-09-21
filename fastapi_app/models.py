from sqlalchemy import Boolean, Column, Integer, String, Numeric
from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Optional
from fastapi_app.database import Base

# Models


"""
class Nodes(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)

    latitude = Column(Numeric(10, 4))
    longitude = Column(Numeric(10, 4))
    area = Column(Numeric(10, 4))
    node_type = Column(String)
    fixed_type = Column(Boolean)
    required_capacity = Column(Numeric(10, 4))
    max_power = Column(Numeric(10, 4))
    is_connected = Column(Boolean)
"""


class Nodes():

    id: int
    lat: float
    long: float
    x: float
    y: float
    area: float
    node_type: str
    peak_demand: float
    is_connected: bool


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
    lat: float
    long: float
    x: float
    y: float
    area: float
    node_type: str
    peak_demand: float
    is_connected: bool


class OptimizeGridRequest(BaseModel):
    price_pole: float
    price_household: float
    price_pole_cable: float
    price_distribution_cable: float
    number_of_relaxation_steps_nr: int
    max_connection_poles: int


class ShsIdentificationRequest(BaseModel):
    cable_price_per_meter_for_shs_mst_identification: float
    connection_cost_to_minigrid: float
    price_shs_hd: float
    price_shs_md: float
    price_shs_ld: float
    algo: str


class SelectBoundariesRequest(BaseModel):
    boundary_coordinates: list


class GenerateExportFileRequest(BaseModel):
    price_pole: float
    price_household: float
    price_pole_cable: float
    price_distribution_cable: float
    shs_identification_cable_price: float
    shs_identification_connection_price: float
    number_of_relaxation_steps_nr: int
