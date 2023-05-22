from datetime import datetime
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import netCDF4
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from feedinlib import era5, Photovoltaic
from shapely.geometry import Point, Polygon
from fastapi_app.io.db.inserts import insert_df
from fastapi_app.io.db import models
from fastapi_app.io.db import queries


def download_weather_data(start_date, end_date, fiel_name='ERA5_weather_data2.nc', country='Nigeria'):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country_shape = world[world['name'] == country]
    geopoints = country_shape.geometry.iloc[0].bounds
    lat = [geopoints[0], geopoints[2]]
    lon = [geopoints[1], geopoints[3]]
    variable = "pvlib"
    era5.get_era5_data_from_datespan_and_position(
        variable=variable,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        latitude=lat, longitude=lon,
        target_file=fiel_name)


async def get_dc_feed_in(lat, lon, start, end):
    weather_df = await queries.get_weather_data(lat, lon, start, end)
    module = pvlib.pvsystem.retrieve_sam('SandiaMod')['SolarWorld_Sunmodule_250_Poly__2013_']
    inverter = pvlib.pvsystem.retrieve_sam('cecinverter')['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    system = PVSystem(surface_tilt=30,
                      surface_azimuth=180,
                      module_parameters=module,
                      inverter_parameters=inverter,
                      temperature_model_parameters=temperature_model_parameters)
    location = Location(latitude=lat, longitude=lon)
    mc = ModelChain(system, location)
    mc.run_model(weather=weather_df)
    dc_power = mc.results.dc['p_mp'].clip(0).fillna(0) / 1000
    return dc_power