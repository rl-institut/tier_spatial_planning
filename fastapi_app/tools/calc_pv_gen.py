from datetime import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import netCDF4
import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from feedinlib import era5, Photovoltaic
from shapely.geometry import Point, Polygon


def download_weather_data(start_date, end_date, fiel_name='ERA5_weather_data.nc', country='Nigeria'):
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


def get_all_locations(file_name='ERA5_weather_data.nc'):
    with netCDF4.Dataset(file_name, 'r') as dataset:
        lat = dataset.variables['latitude'][:]
        lon = dataset.variables['longitude'][:]
    lon_grid, lat_grid = np.meshgrid(lat, lon)
    grid_points = np.stack((lat_grid, lon_grid), axis=-1)
    grid_points = grid_points.reshape(-1, 2)
    return grid_points


def get_weather_data(lat, lon, file_name='ERA5_weather_data.nc'):
    df = era5.weather_df_from_era5(file_name, "pvlib", area=[lon, lat])
    df.index = df.index - pd.Timedelta('30min')
    solar_position = pvlib.solarposition.get_solarposition(
        time=df.index,
        latitude=lat,
        longitude=lon    )
    df['dni'] = pvlib.irradiance.disc(ghi=df['ghi'],
                                          datetime_or_doy=df.index,
                                          solar_zenith=solar_position['apparent_zenith'])['dni']
    return df


def get_dc_feed_in(df, lat, lon,):
    module = pvlib.pvsystem.retrieve_sam('SandiaMod')['Canadian_Solar_CS5P_220M___2009_']
    inverter = pvlib.pvsystem.retrieve_sam('cecinverter')['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    system = PVSystem(surface_tilt=30,
                      surface_azimuth=180,
                      module_parameters=module,
                      inverter_parameters=inverter,
                      temperature_model_parameters=temperature_model_parameters)
    location = Location(latitude=lat, longitude=lon)
    mc = ModelChain(system, location)
    mc.run_model(weather=df)

    # The resulting timeseries of DC power output is stored in the ac attribute
    dc_power = mc.results.dc['p_mp'].clip(0) / 1000
    return dc_power

if __name__ == '__main__':
    start_date = pd.Timestamp(datetime.date(datetime.now()), tz='Africa/Lagos')
    end_date = start_date + pd.Timedelta(days=365)
    grid_points = get_all_locations()
    lat, lon = grid_points[0]
    df = get_weather_data(lat, lon)
    dc_ps = get_dc_feed_in(df, lat, lon)
    t = 4

