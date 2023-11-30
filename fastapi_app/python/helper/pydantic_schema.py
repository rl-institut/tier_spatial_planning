import inspect
from typing import List, Dict, Union, Optional

import flatten_dict
import pandas as pd
from flatten_dict.reducers import make_reducer
from pydantic import BaseModel


class Credentials(BaseModel):
    email: str
    password: str
    remember_me: bool


class ChangePW(BaseModel):
    new_password: str
    old_password: str
    captcha_input: str
    captcha_hash: str


class ValidRegistration(BaseModel):
    validation: bool
    msg: str


class Mail(BaseModel):
    body: str
    subject: str
    from_address: str


class Password(BaseModel):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class UserOverview(BaseModel):
    email: str
    project_name: Optional[str] = None


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


class TaskInfo(BaseModel):
    project_id: str
    task_id: str
    time: int
    model: str


class TaskID(BaseModel):
    task_id: str
    project_id: str


class ProjectID(BaseModel):
    project_id: Optional[str] = None


class ImportFileRequestList(BaseModel):
    data: List[ImportFileRequest]


class HasCookies(BaseModel):
    access_token: bool
    consent_cookie: bool


class Email(BaseModel):
    email: str


class MapDataRequest(BaseModel):
    map_elements: list


class MapData(BaseModel):
    boundary_coordinates: list
    map_elements: list


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


class SaveProjectSetup(BaseModel):
    page_setup: Dict[str, str]


class SaveGridDesign(BaseModel):
    grid_design: Dict[str, str]


class SaveDemandEstimation(BaseModel):
    demand_estimation: Dict[str, str]


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

    def to_dict(self):
        attr_dict = dict()
        nested_dict = False
        for (key, value) in inspect.getmembers(self):
            if key[:1] != '_':
                if key.lower() not in ['metadata', 'registry', 'config'] and not inspect.ismethod(value):
                    attr_dict[key] = value
                    if nested_dict is False and isinstance(value, dict):
                        nested_dict = True
        if nested_dict is True:
            attr_dict = flatten_dict.flatten(attr_dict, reducer=make_reducer('__'))
        return attr_dict

    def to_df(self):
        attr_dict = self.to_dict()
        df = pd.DataFrame.from_dict(attr_dict, orient='index').T
        return df
