from sqlalchemy import Boolean, Column, Integer, String, Numeric
# from sqlalchemy.orm import relationship
from pydantic import BaseModel
from fastapi_app.database import Base
from typing import List, Dict, Union

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
    lon_from = Column(Numeric(10, 5))
    lat_to = Column(Numeric(10, 5))
    lon_to = Column(Numeric(10, 5))
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
    node_type: str
    consumer_type: str
    consumer_detail: str
    surface_area: float
    peak_demand: float
    average_consumption: float
    is_connected: bool
    how_added: str


class SavePreviousDataRequest(BaseModel):
    page_setup: Dict[str, str]
    consumer_selection: Dict[str, str]


class OptimizeGridRequest(BaseModel):
    optimization: Dict[str, int]
    constraints: Dict[str, int]


class OptimizeEnergySystemRequest(BaseModel):
    pv: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    diesel_genset: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    battery: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    inverter: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    rectifier: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    shortage: Dict[str, Union[Dict[str, bool], Dict[str, float]]]
    # path_data: str


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
    cost_pole: float
    cost_connection: float
    cost_interpole_cable: float
    cost_distribution_cable: float
    shs_identification_cable_cost: float
    shs_identification_connection_cost: float
    number_of_relaxation_steps_nr: int


class ImportFileRequest(BaseModel):
    latitude: float
    longitude: float


class ImportFileRequestList(BaseModel):
    data: List[ImportFileRequest]
