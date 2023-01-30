import pandas as pd
import sqlalchemy as sa
from sqlalchemy import Boolean, Column, Integer, VARCHAR, Numeric, String, DateTime, SMALLINT, FLOAT
from sqlalchemy.dialects.mysql import DECIMAL, TINYINT
# from sqlalchemy.orm import relationship
from fastapi_app.db.database import Base
from typing import List, Dict, Union
from pydantic import BaseModel, EmailStr


class Credentials(BaseModel):
    email: str
    password: str


class ValidRegistration(BaseModel):
    validation: bool
    msg: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserOverview(BaseModel):
    email: str


class User(Base):
    id = Column(SMALLINT, primary_key=True, index=True)
    email = Column(VARCHAR(255), nullable=False, unique=True, index=True)
    hashed_password = Column(VARCHAR(255), nullable=False)
    guid = Column(VARCHAR(12), nullable=False)
    is_confirmed = Column(Boolean(), default=False)
    is_active = Column(Boolean(), default=False)
    is_superuser = Column(Boolean(), default=False)


class ProjectSetup(Base):
    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, unique=False, index=True)
    name = Column(VARCHAR(61), nullable=True, unique=False)
    descr = Column(VARCHAR(201), nullable=True, unique=False)
    country = Column(VARCHAR(50), nullable=True, unique=False)
    state = Column(VARCHAR(50), nullable=True, unique=False)
    created_at = Column(DateTime, nullable=False, server_default=sa.func.now())
    updated_at = Column(DateTime, nullable=False, server_default=sa.func.now(), server_onupdate=sa.func.now(),)
    interest_rate = Column(FLOAT, nullable=False)
    project_lifetime = Column(TINYINT, nullable=False, server_default="25")
    start_date = Column(DateTime, nullable=False, default=pd.to_datetime(str(pd.Timestamp.now().year - 1)))
    temporal_resolution = Column(SMALLINT, nullable=False, server_default="1")
    number_of_days = Column(SMALLINT, nullable=False, server_default="365")


class Nodes(Base):
    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Numeric(10, 5))
    longitude = Column(Numeric(10, 5))
    x = Column(Numeric(10, 5))
    y = Column(Numeric(10, 5))
    area = Column(Numeric(10, 2))
    node_type = Column(VARCHAR(55))
    peak_demand = Column(Numeric(10, 3))
    is_connected = Column(Boolean)
    how_added = Column(VARCHAR(55))


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
    id = Column(Integer, primary_key=True, index=True)
    lat_from = Column(Numeric(10, 5))
    lon_from = Column(Numeric(10, 5))
    lat_to = Column(Numeric(10, 5))
    lon_to = Column(Numeric(10, 5))
    x_from = Column(Numeric(10, 5))
    y_from = Column(Numeric(10, 5))
    x_to = Column(Numeric(10, 5))
    y_to = Column(Numeric(10, 5))
    link_type = Column(VARCHAR(55))
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
    grid_design: Dict[str, str]


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
    cost_distribution_cable: float
    cost_connection_cable: float
    shs_identification_cable_cost: float
    shs_identification_connection_cost: float
    number_of_relaxation_steps_nr: int


class ImportFileRequest(BaseModel):
    latitude: float
    longitude: float


class ImportFileRequestList(BaseModel):
    data: List[ImportFileRequest]
