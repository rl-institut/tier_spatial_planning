import sqlalchemy as sa
import flatten_dict
from flatten_dict.reducers import make_reducer
from sqlalchemy import Boolean, Column, VARCHAR, Numeric, DateTime
from sqlalchemy.dialects.mysql import TINYINT, SMALLINT, FLOAT
import json
import inspect
import pandas as pd
from typing import Any
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy.dialects.mysql.json import JSON



@as_declarative()
class Base:
    id: Any
    __name__: str

    # generate tablename from classname
    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

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

    def to_json(self):
        df = self.to_df().dropna(how='all', axis=0)
        data_json = json.loads(df.to_json())
        return data_json


class User(Base):

    @staticmethod
    def __name__():
        return 'User'

    id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    email = Column(VARCHAR(100), nullable=False, unique=True, index=True)
    hashed_password = Column(VARCHAR(255), nullable=False)
    guid = Column(VARCHAR(45), nullable=False)
    is_confirmed = Column(Boolean(), default=False)
    is_active = Column(Boolean(), default=False)
    is_superuser = Column(Boolean(), default=False)
    task_id = Column(VARCHAR(80))
    project_id = Column(SMALLINT(unsigned=True), default=False, nullable=True)
    show_tutorial = Column(SMALLINT(unsigned=True), default=True, nullable=True)


class ProjectSetup(Base):

    @staticmethod
    def __name__():
        return 'ProjectSetup'

    id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    project_id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    project_name = Column(VARCHAR(51), nullable=True, unique=False)
    project_description = Column(VARCHAR(201), nullable=True, unique=False)
    created_at = Column(DateTime, nullable=False, server_default=sa.func.now())
    updated_at = Column(DateTime, nullable=False, server_default=sa.func.now(), server_onupdate=sa.func.now(),)
    interest_rate = Column(FLOAT(unsigned=True), nullable=False)
    project_lifetime = Column(TINYINT(unsigned=True), nullable=False, server_default="25")
    start_date = Column(DateTime, nullable=False, default=pd.to_datetime(str(pd.Timestamp.now().year - 1)))
    temporal_resolution = Column(SMALLINT(unsigned=True), nullable=False, server_default="1")
    n_days = Column(SMALLINT(unsigned=True), nullable=False, server_default="365")
    status = Column(VARCHAR(25), default="not yet started")
    email_notification = Column(Boolean(), default=False)


class GridDesign(Base):

    @staticmethod
    def __name__():
        return 'GridDesign'

    id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    project_id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    distribution_cable_lifetime = Column(TINYINT(unsigned=True))
    distribution_cable_capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    distribution_cable_max_length = Column(TINYINT(unsigned=True))
    connection_cable_lifetime = Column(TINYINT(unsigned=True))
    connection_cable_capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    connection_cable_max_length = Column(TINYINT(unsigned=True))
    pole_lifetime = Column(TINYINT(unsigned=True))
    pole_capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    pole_max_n_connections = Column(TINYINT(unsigned=True))
    mg_connection_cost = Column(FLOAT(precision=5, scale=1, unsigned=True))
    shs_max_grid_cost = Column(FLOAT(precision=5, scale=1, unsigned=True))

class EnergySystemDesign(Base):

    @staticmethod
    def __name__():
        return 'EnergySystemDesign'

    id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    project_id = Column(SMALLINT(unsigned=True), primary_key=True, index=True)
    battery__settings__is_selected = Column(Boolean(), default=False)
    battery__settings__design = Column(Boolean(), default=False)
    battery__parameters__nominal_capacity = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__lifetime = Column(TINYINT(unsigned=True))
    battery__parameters__capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__opex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__soc_min = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__soc_max = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__c_rate_in = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__c_rate_out = Column(FLOAT(precision=5, scale=1, unsigned=True))
    battery__parameters__efficiency = Column(FLOAT(precision=5, scale=2, unsigned=True))
    diesel_genset__settings__is_selected = Column(Boolean(), default=False)
    diesel_genset__settings__design = Column(Boolean(), default=False)
    diesel_genset__parameters__nominal_capacity = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__lifetime = Column(TINYINT(unsigned=True))
    diesel_genset__parameters__capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__opex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__variable_cost = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__fuel_cost = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__fuel_lhv = Column(FLOAT(precision=5, scale=1, unsigned=True))
    diesel_genset__parameters__min_load = Column(FLOAT(precision=5, scale=2, unsigned=True))
    diesel_genset__parameters__max_load = Column(FLOAT(precision=5, scale=2, unsigned=True))
    diesel_genset__parameters__min_efficiency = Column(FLOAT(precision=5, scale=2, unsigned=True))
    diesel_genset__parameters__max_efficiency = Column(FLOAT(precision=5, scale=2, unsigned=True))
    inverter__settings__is_selected = Column(Boolean(), default=False)
    inverter__settings__design = Column(Boolean(), default=False)
    inverter__parameters__nominal_capacity = Column(FLOAT(precision=5, scale=1, unsigned=True))
    inverter__parameters__lifetime = Column(TINYINT(unsigned=True))
    inverter__parameters__capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    inverter__parameters__opex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    inverter__parameters__efficiency = Column(FLOAT(precision=5, scale=2, unsigned=True))
    pv__settings__is_selected = Column(Boolean(), default=False)
    pv__settings__design = Column(Boolean(), default=False)
    pv__parameters__nominal_capacity = Column(FLOAT(precision=5, scale=1, unsigned=True))
    pv__parameters__lifetime = Column(TINYINT(unsigned=True))
    pv__parameters__capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    pv__parameters__opex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    rectifier__settings__is_selected = Column(Boolean(), default=False)
    rectifier__settings__design = Column(Boolean(), default=False)
    rectifier__parameters__nominal_capacity = Column(FLOAT(precision=5, scale=1, unsigned=True))
    rectifier__parameters__lifetime = Column(TINYINT(unsigned=True))
    rectifier__parameters__capex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    rectifier__parameters__opex = Column(FLOAT(precision=5, scale=1, unsigned=True))
    rectifier__parameters__efficiency = Column(FLOAT(precision=5, scale=2, unsigned=True))
    shortage__settings__is_selected = Column(FLOAT(precision=5, scale=1, unsigned=True))
    shortage__parameters__max_shortage_total = Column(FLOAT(precision=5, scale=3, unsigned=True))
    shortage__parameters__max_shortage_timestep = Column(FLOAT(precision=5, scale=3, unsigned=True))
    shortage__parameters__shortage_penalty_cost = Column(FLOAT(precision=5, scale=2, unsigned=True))


class Nodes(Base):

    @staticmethod
    def __name__():
        return 'Nodes'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    latitude = Column(Numeric(9, 6), primary_key=True)
    longitude = Column(Numeric(9, 6), primary_key=True)
    node_type = Column(VARCHAR(55), primary_key=True)
    consumer_type = Column(VARCHAR(55))
    consumer_detail = Column(VARCHAR(55))
    is_connected = Column(Boolean)
    how_added = Column(VARCHAR(55))
    distance_to_load_center = Column(Numeric(10, 6))
    parent = Column(VARCHAR(10))
    distribution_cost = Column(Numeric(10, 5))
    custom_specification = Column(VARCHAR(1000))
    shs_options = Column(SMALLINT)


class Links(Base):

    @staticmethod
    def __name__():
        return 'Links'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    lat_from = Column(Numeric(9, 6), primary_key=True)
    lon_from = Column(Numeric(9, 6), primary_key=True)
    lat_to = Column(Numeric(9, 6), primary_key=True)
    lon_to = Column(Numeric(9, 6), primary_key=True)
    link_type = Column(VARCHAR(50), primary_key=True)
    length = Column(SMALLINT(unsigned=True))


class Demand(Base):

    @staticmethod
    def __name__():
        return 'Demand'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    household_option = Column(SMALLINT, default=0)
    maximum_peak_load = Column(Numeric(10, 3), default=None)
    average_daily_energy = Column(Numeric(10, 3), default=None)


class Results(Base):

    @staticmethod
    def __name__():
        return 'Results'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    n_consumers = Column(SMALLINT(unsigned=True))
    n_shs_consumers = Column(SMALLINT(unsigned=True))
    n_poles = Column(SMALLINT(unsigned=True))
    n_distribution_links = Column(SMALLINT(unsigned=True))
    n_connection_links = Column(SMALLINT(unsigned=True))
    length_distribution_cable = Column(SMALLINT(unsigned=True))
    average_length_distribution_cable = Column(Numeric(10, 3))
    length_connection_cable = Column(SMALLINT(unsigned=True))
    average_length_connection_cable = Column(Numeric(10, 3))
    cost_grid = Column(SMALLINT(unsigned=True))
    cost_shs = Column(SMALLINT(unsigned=True))
    lcoe = Column(SMALLINT(unsigned=True))
    res = Column(Numeric(10, 3))
    shortage_total = Column(Numeric(10, 3))
    surplus_rate = Column(Numeric(10, 3))
    cost_renewable_assets = Column(Numeric(10, 3))
    cost_non_renewable_assets = Column(Numeric(10, 3))
    cost_fuel = Column(Numeric(10, 3))
    pv_capacity = Column(Numeric(10, 3))
    battery_capacity = Column(Numeric(10, 3))
    inverter_capacity = Column(Numeric(10, 3))
    rectifier_capacity = Column(Numeric(10, 3))
    diesel_genset_capacity = Column(Numeric(10, 3))
    peak_demand = Column(Numeric(10, 3))
    surplus = Column(Numeric(10, 3))
    fuel_to_diesel_genset = Column(Numeric(10, 3))
    diesel_genset_to_rectifier = Column(Numeric(10, 3))
    diesel_genset_to_demand = Column(Numeric(10, 3))
    rectifier_to_dc_bus = Column(Numeric(10, 3))
    pv_to_dc_bus = Column(Numeric(10, 3))
    battery_to_dc_bus = Column(Numeric(10, 3))
    dc_bus_to_battery = Column(Numeric(10, 3))
    dc_bus_to_inverter = Column(Numeric(10, 3))
    dc_bus_to_surplus = Column(Numeric(10, 3))
    inverter_to_demand = Column(Numeric(10, 3))
    time_grid_design = Column(Numeric(10, 3))
    time_energy_system_design = Column(Numeric(10, 3))
    time = Column(Numeric(10, 3))
    co2_savings = Column(Numeric(10, 3))
    max_voltage_drop = Column(Numeric(3, 1))
    infeasible = Column(TINYINT(unsigned=True))
    average_annual_demand_per_consumer = Column(Numeric(10, 1))
    total_annual_consumption = Column(Numeric(10, 1))
    upfront_invest_grid = Column(Numeric(10, 1))
    upfront_invest_diesel_gen = Column(Numeric(10, 1))
    upfront_invest_inverter = Column(Numeric(10, 1))
    upfront_invest_rectifier = Column(Numeric(10, 1))
    upfront_invest_battery = Column(Numeric(10, 1))
    upfront_invest_pv = Column(Numeric(10, 1))
    co2_emissions = Column(Numeric(10, 1))
    fuel_consumption = Column(Numeric(10, 1))
    base_load = Column(Numeric(10, 1))
    max_shortage =  Column(Numeric(10, 1))
    epc_total =  Column(Numeric(10, 1))
    epc_pv = Column(Numeric(10, 1))
    epc_diesel_genset = Column(Numeric(10, 1))
    epc_inverter = Column(Numeric(10, 1))
    epc_rectifier = Column(Numeric(10, 1))
    epc_battery = Column(Numeric(10, 1))


class DemandCoverage(Base):

    @staticmethod
    def __name__():
        return 'DemandCoverage'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    data = Column(JSON)


class EnergyFlow(Base):

    @staticmethod
    def __name__():
        return 'EnergyFlow'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    data = Column(JSON)


class DurationCurve(Base):

    @staticmethod
    def __name__():
        return 'DurationCurve'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    data = Column(JSON)


class Emissions(Base):

    @staticmethod
    def __name__():
        return 'Emissions'

    id = Column(SMALLINT, primary_key=True, index=True)
    project_id = Column(SMALLINT, primary_key=True, index=True)
    data = Column(JSON)


class WeatherData(Base):

    @staticmethod
    def __name__():
        return 'WeatherData'

    dt = Column(DateTime, primary_key=True, index=True)
    lat = Column(Numeric(7, 3), primary_key=True, index=True)
    lon = Column(Numeric(10, 7), primary_key=True, index=True)
    wind_speed = Column(Numeric(5, 2))
    temp_air = Column(Numeric(4, 2))
    ghi = Column(Numeric(6, 2))
    dni = Column(Numeric(6, 2))
    dhi = Column(Numeric(6, 2))
