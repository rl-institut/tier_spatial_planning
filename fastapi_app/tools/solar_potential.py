import warnings
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
from fastapi_app.db import  sync_queries, async_queries
from fastapi_app import config


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    df['dni'] = np.nan
    grid_points = retrieve_grid_points(data_xr)
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
    df.iloc[:, 3:] = (df.iloc[:, 3:] + 0.0000001).round(1)
    df.loc[:, 'lon'] = df.loc[:, 'lon'].round(3)
    df.loc[:, 'lat'] = df.loc[:, 'lat'].round(7)
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(str)
    return df


def retrieve_grid_points(ds):
    lat = ds.variables['latitude'][:]
    lon = ds.variables['longitude'][:]
    lon_grid, lat_grid = np.meshgrid(lat, lon)
    grid_points = np.stack((lat_grid, lon_grid), axis=-1)
    grid_points = grid_points.reshape(-1, 2)
    return grid_points


def get_closest_grid_point(lat, lon):
    lats = pd.Series([14.442, 14.192, 13.942, 13.692, 13.442, 13.192, 12.942, 12.692, 12.442,
                      12.192, 11.942, 11.692, 11.442, 11.192, 10.942, 10.692, 10.442, 10.192,
                      9.942, 9.692, 9.442, 9.192, 8.942, 8.692, 8.442, 8.192, 7.942,
                      7.692, 7.442, 7.192, 6.942, 6.692, 6.442, 6.192, 5.942, 5.692,
                      5.442, 5.192, 4.942, 4.692, 4.442, 4.192, 3.942, 3.692, 3.442,
                      3.192, 2.942, 2.691])
    lons = pd.Series([4.24, 4.490026, 4.740053, 4.990079, 5.240105, 5.490131,
                      5.740158, 5.990184, 6.240211, 6.490237, 6.740263, 6.99029,
                      7.240316, 7.490342, 7.740368, 7.990395, 8.240421, 8.490447,
                      8.740474, 8.9905, 9.240526, 9.490553, 9.740579, 9.990605,
                      10.240631, 10.490658, 10.740685, 10.99071, 11.240737, 11.490763,
                      11.740789, 11.990816, 12.240842, 12.490869, 12.740894, 12.990921,
                      13.240948, 13.490973, 13.741])
    closest_lat = round(lats.loc[(lats - lat).abs().idxmin()], 3)
    closest_lon = round(lons.loc[(lons - lon).abs().idxmin()], 3)
    return closest_lat, closest_lon



async def get_dc_feed_in(lat, lon, start, end):
    weather_df = await async_queries.get_weather_data(lat, lon, start, end)
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

