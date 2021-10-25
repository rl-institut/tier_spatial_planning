from sqlalchemy import Boolean, Column, Integer, String, Numeric
#from sqlalchemy.orm import relationship
from pydantic import BaseModel
from typing import Optional
from fastapi_app.database import Base

# Models


class Nodes(Base):
    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True, index=True)

    latitude = Column(Numeric(10, 5))
    longitude = Column(Numeric(10, 5))
    x = Column(Numeric(10, 5))
    y = Column(Numeric(10, 5))
    area = Column(Numeric(10, 2))
    node_type = Column(String)
    peak_demand = Column(Numeric(10, 3))
    is_connected = Column(Boolean)
    how_added = Column(String)


# class Nodes():

#     id: int
#     lat: float
#     long: float
#     x: float
#     y: float
#     area: float
#     node_type: str
#     peak_demand: float
#     is_connected: bool


class Links(Base):
    __tablename__ = "links"

    id = Column(Integer, primary_key=True, index=True)
    lat_from = Column(Numeric(10, 5))
    long_from = Column(Numeric(10, 5))
    lat_to = Column(Numeric(10, 5))
    long_to = Column(Numeric(10, 5))
    x_from = Column(Numeric(10, 5))
    y_from = Column(Numeric(10, 5))
    x_to = Column(Numeric(10, 5))
    y_to = Column(Numeric(10, 5))
    link_type = Column(String)
    cable_thickness = Column(Numeric(10, 3))
    length = Column(Numeric(10, 2))


class AddNodeRequest(BaseModel):
    latitude: float
    longitude: float
    area: float
    node_type: str
    consumer_type: str
    demand_type: str
    peak_demand: float
    is_connected: bool
    how_added: str


class OptimizeGridRequest(BaseModel):
    price_pole: float
    price_consumer: float
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
    price_consumer: float
    price_pole_cable: float
    price_distribution_cable: float
    shs_identification_cable_price: float
    shs_identification_connection_price: float
    number_of_relaxation_steps_nr: int
