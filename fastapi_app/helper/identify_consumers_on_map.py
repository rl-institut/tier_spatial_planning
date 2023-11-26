import numpy as np
import urllib.request
import datetime
import time
import json
import math
from shapely import geometry

def get_consumer_within_boundaries(df):
    # min and max of latitudes and longitudes are sent to the overpass to get
    # a large rectangle including (maybe) more buildings than selected
    min_latitude, min_longitude, max_latitude, max_longitude \
        = df['latitude'].min(), df['longitude'].min(), df['latitude'].max(),  df['longitude'].max()
    url = (f'https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500]'
           f'[bbox:{min_latitude},{min_longitude},{max_latitude},{max_longitude}];'
           f'way["building"="yes"];(._;>;);out;')
    url_formated = url.replace(" ", "+")
    with urllib.request.urlopen(url_formated) as url:
        res = url.read().decode()
        if len(res) > 0:
            data = json.loads(res)
        else:
            return None, None
    # first converting the json file, which is delievered by overpass to geojson,
    # then obtaining coordinates and surface areas of all buildings inside the
    # 'big' rectangle.
    geojson_data = convert_overpass_json_to_geojson(data)

    building_coord, building_surface_areas = obtain_areas_and_mean_coordinates_from_geojson(geojson_data)
    # excluding the buildings which are outside the drawn boundary
    mask_building_within_boundaries = {key: is_point_in_boundaries(value, df.values.tolist())
                                       for key, value in building_coord.items()}
    building_coordidates_within_boundaries = \
        {key: value for key, value in building_coord.items() if mask_building_within_boundaries[key]}
    return data, building_coordidates_within_boundaries


def convert_overpass_json_to_geojson(json_dict):
    """
    This function convert dict obtained using the overpass api into
    a GEOJSON dict containing only the polygons of the buildings.

    Parameters
    ----------
    json_dict (dict):
        dict obtained using the overpass api.
        Example: json at https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500][bbox:11.390617069027885,9.132004976127066,11.392010636465772,9.133802056167044];(way["building"];relation["building"];);out body;>;out skel qt;

    """
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

    node_coordinates = {element["id"]:  [element["lat"], element["lon"]]
                        for element in json_dict["elements"] if element["type"] == "node"}

    geojson = {
        "type": "FeatureCollection",
        "generator": "overpass-ide, formated by PeopleSun WP4 Tool",
        "timestamp": timestamp,
        "features": [
            {
                "type": "Feature",
                "property": {
                    "@id": f"{d['type']}/{d['id']}",
                    "building": "yes",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            node_coordinates[node] for node in d["nodes"]
                        ],
                    ],
                },
            } for d in json_dict['elements'] if d['type'] == "way"]}
    return geojson


def obtain_areas_and_mean_coordinates_from_geojson(geojson: dict):

    building_mean_coordinates = {}
    building_surface_areas = {}

    if len(geojson["features"]) != 0:
        reference_coordinate = geojson["features"][0]["geometry"]["coordinates"][0][0]
        for building in geojson["features"]:
            xy_coordinates = []
            latitudes_longitudes = [coord for coord in building["geometry"]["coordinates"][0]]
            latitudes = [x[0] for x in latitudes_longitudes]
            longitudes = [x[1] for x in latitudes_longitudes]
            mean_coord = [np.mean(latitudes), np.mean(longitudes)]
            for edge in range(len(latitudes)):
                xy_coordinates.append(xy_coordinates_from_latitude_longitude(
                    latitude=latitudes_longitudes[edge][0],
                    longitude=latitudes_longitudes[edge][1],
                    ref_latitude=reference_coordinate[0],
                    ref_longitude=reference_coordinate[1]))
            polygon = geometry.Polygon(xy_coordinates)
            area = polygon.area
            perimeter = polygon.length
            compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter else 0
            if area > 4 \
                    and not (0.81 < compactness < 1.19 and area < 8):
                building_mean_coordinates[building["property"]["@id"]] = mean_coord
                building_surface_areas[building["property"]["@id"]] = area
    return building_mean_coordinates, building_surface_areas


def obtain_mean_coordinates_from_geojson(df):
    """
    This function creates a dictionnary with the 'id' of each building as a key
    and the mean loaction of the building as value in the form [lat, long].

    Parameters
    ----------
        geojson (dict):
            Dictionary containing the geojson data of the building of a
            specific area. Output of the
            tools.conversion.convert_overpass_json_to_geojson function.

    Returns
    -------
        Dict containing the 'id' of each building as a key
        and the mean loaction of the building as value in the form [long, lat].

        Dict containing the 'id' of each building as a key
    """
    if not df.empty:
        df1 = df[df['type'] == 'way']
        df2 = df[df['type'] == 'node'].set_index('id')

        df2['lat_lon'] = list(zip(df2['lat'], df2['lon']))
        index_to_lat_lon = df2['lat_lon'].to_dict()
        df1_exploded = df1.explode('nodes')
        df1_exploded['nodes'] = df1.explode('nodes')['nodes'].map(index_to_lat_lon)
        df1['nodes'] = df1_exploded.groupby(df1_exploded.index).agg({'nodes': list})
        building_mean_coordinates = {}
        if not df1.empty:
            for row_idx, row in df1.iterrows():
                latitudes_longitudes = [coord for coord in row["nodes"]]
                latitudes = [x[0] for x in latitudes_longitudes]
                longitudes = [x[1] for x in latitudes_longitudes]
                mean_coord = [np.mean(latitudes), np.mean(longitudes)]
                building_mean_coordinates[row["id"]] = mean_coord
        return building_mean_coordinates
    else:
        return {}, {}


def is_point_in_boundaries(point_coordinates: tuple,
                           boundaries: tuple):
    """ 
    Function that checks whether or not 2D point lies within boundaries

    Parameter
    ---------
    coordinates (list or tuple):
        Coordinates of the point in format [x, y]

    boundaries (list or tuple):
        Coordinates of the angle of the polygon forming the boundaries in format
        [[x1, y1], [x2, y2], ..., [xn, yn]] for a polygon with n vertices.
 """
    polygon = geometry.Polygon(boundaries)
    point = geometry.Point(point_coordinates)

    return polygon.contains(point)


def are_points_in_boundaries(df, boundaries):
    polygon = geometry.Polygon(boundaries)
    df['inside_boundary'] = df.apply(lambda row: polygon.contains(geometry.Point([row['latitude'], row['longitude']])),
                                     axis=1)
    return df['inside_boundary']


def xy_coordinates_from_latitude_longitude(latitude, longitude, ref_latitude, ref_longitude):
    """ This function converts (latitude, longitude) coordinates into (x, y)
    plane coordinates using a reference latitude and longitude.

    Parameters
    ----------
        latitude (float):
            Latitude (in degree) to be converted.

        longitude (float):
            Longitude (in degree) to be converted.

        ref_latitude (float):
            Reference latitude (in degree).

        ref_longitude (float):
            Reference longitude (in degree).

    Return
    ------
        (tuple):
            (x, y) plane coordinates.
    """

    r = 6371000     # Radius of the earth [m]
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)
    ref_latitude_rad = math.radians(ref_latitude)
    ref_longitude_rad = math.radians(ref_longitude)

    x = r * (longitude_rad - ref_longitude_rad) * math.cos(ref_latitude)
    y = r * (latitude_rad - ref_latitude_rad)
    return x, y
