import os
import pandas as pd
import numpy as np
import geopandas as gpd
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from feedinlib import era5
from fastapi_app.db import queries, sync_queries, config


def create_cdsapirc_file():
    home_dir = os.path.expanduser('~')
    file_path = os.path.join(home_dir, '.cdsapirc')
    if os.path.exists(file_path):
        print(f".cdsapirc file already exists at {file_path}")
        return
    content = f"url: https://cds.climate.copernicus.eu/api/v2\nkey: {config.CDS_API_KEY}"
    with open(file_path, 'w') as file:
        file.write(content)
    print(f".cdsapirc file created at {file_path}")


def download_weather_data(start_date, end_date, country='Nigeria', target_file='file'):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    country_shape = world[world['name'] == country]
    geopoints = country_shape.geometry.iloc[0].bounds
    lat = [geopoints[0], geopoints[2]]
    lon = [geopoints[1], geopoints[3]]
    variable = "pvlib"
    create_cdsapirc_file()
    data_xr = era5.get_era5_data_from_datespan_and_position(
                variable=variable,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                latitude=lat,
                longitude=lon,
                target_file=target_file)
    return data_xr


def prepare_weather_data(data_xr):
    df = era5.format_pvlib(data_xr)
    df = df.reset_index()
    df = df.rename(columns={'time': 'dt', 'latitude': 'lat', 'longitude': 'lon'})
    df = df.set_index(['dt'])
    def get_all_locations(ds):
        lat = ds.variables['latitude'][:]
        lon = ds.variables['longitude'][:]
        lon_grid, lat_grid = np.meshgrid(lat, lon)
        grid_points = np.stack((lat_grid, lon_grid), axis=-1)
        grid_points = grid_points.reshape(-1, 2)
        return grid_points
    df['dni'] = np.nan
    grid_points = get_all_locations(data_xr)
    for lon, lat in grid_points:
        mask = (df['lat'] == lat) & (df['lon'] == lon)
        tmp_df = df.loc[mask]
        solar_position = pvlib.solarposition.get_solarposition(time=tmp_df.index,
                                                               latitude=lat,
                                                               longitude=lon)
        df.loc[mask, 'dni'] = pvlib.irradiance.dni(ghi=tmp_df['ghi'],
                                                   dhi=tmp_df['dhi'],
                                                   zenith=solar_position['apparent_zenith']).fillna(0)
    df = df.reset_index()
    df['dt'] = df['dt'] - pd.Timedelta('30min')
    df['dt'] = df['dt'].dt.tz_convert('UTC').dt.tz_localize(None)
    df.iloc[:, 3:] = df.iloc[:, 3:] + 0.0000001
    df.iloc[:, 3:] = df.iloc[:, 3:].round(1)
    df.loc[:, 'lon'] = df.loc[:, 'lon'].round(3)
    df.loc[:, 'lat'] = df.loc[:, 'lat'].round(7)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(str)
    return df



async def get_dc_feed_in(lat, lon, start, end):
    weather_df = await queries.get_weather_data(lat, lon, start, end)
    return _get_dc_feed_in(lat, lon, weather_df)

def get_dc_feed_in_sync_db_query(lat, lon, start, end):
    weather_df = sync_queries.get_weather_data(lat, lon, start, end)
    return _get_dc_feed_in(lat, lon, weather_df)


def _get_dc_feed_in(lat, lon, weather_df):
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

