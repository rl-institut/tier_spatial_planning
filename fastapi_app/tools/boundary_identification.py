import numpy as np
import pandas as pd
import urllib.request
import json

import fastapi_app.tools.coordinates_conversion as conv
from shapely import geometry


def get_consumer_within_boundaries(df):
    # min and max of latitudes and longitudes are sent to the overpass to get
    # a large rectangle including (maybe) more buildings than selected
    min_latitude, min_longitude, max_latitude, max_longitude \
        = df['latitude'].min(), df['longitude'].min(), df['latitude'].max(),  df['longitude'].max()
    url = f'https://www.overpass-api.de/api/interpreter?data=[out:json][timeout:2500]' \
          f'[bbox:{min_latitude},{min_longitude},{max_latitude},{max_longitude}];' \
          f'(way["building"="yes"];relation["building"];);out body;>;out skel qt;'
    url_formated = url.replace(" ", "+")
    with urllib.request.urlopen(url_formated) as url:
        res = url.read().decode()
        if len(res) > 0:
            data = json.loads(res)
        else:
            return None, None, None
    # first converting the json file, which is delievered by overpass to geojson,
    # then obtaining coordinates and surface areas of all buildings inside the
    # 'big' rectangle.
    df2 = pd.DataFrame.from_dict(data['elements'])
    if df2.empty:
        return None, None, None
    building_coord = obtain_mean_coordinates_from_geojson(df2)
    # excluding the buildings which are outside the drawn boundary
    mask_building_within_boundaries = {key: is_point_in_boundaries(value, df.values.tolist())
                                       for key, value in building_coord.items()}
    building_coordidates_within_boundaries = \
        {key: value for key, value in building_coord.items() if mask_building_within_boundaries[key]}
    return data, building_coordidates_within_boundaries


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
    retrieve_building_area_from_overpass = False
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
            reference_coordinate = df1['nodes'].iloc[0][0]
            for row_idx, row in df1.iterrows():
                xy_coordinates = []
                latitudes_longitudes = [coord for coord in row["nodes"]]
                latitudes = [x[0] for x in latitudes_longitudes]
                longitudes = [x[1] for x in latitudes_longitudes]
                mean_coord = [np.mean(latitudes), np.mean(longitudes)]

                building_mean_coordinates[row["id"]] = mean_coord
        return building_mean_coordinates
    else:
        return {}, {}


def are_segments_crossing(segment1, segment2):
    """
    Function that checks weather two 2D segments are crossing/intersecting.
    Inspired from https://algorithmtutor.com/Computational-Geometry/Check-if-two-line-segment-intersect/

    Parameters
    ----------
    segment1 (list or tuple):
        coordinates of the two end points of the first segment in format
        ((x1, y1), (x2, y2))

    segment2 (list or tuple):
        coordinates of the two end points of the second segment in format
        ((x1, y1), (x2, y2))

    Output
    ------
        Returns True is the two segments are intersecting, otherwise returns
        False

    Notes
    -----
    If the two segments are just touching without intersecting, the function
    return False
    """

    p1 = np.array(segment1[0])
    p2 = np.array(segment1[1])
    p3 = np.array(segment2[0])
    p4 = np.array(segment2[1])

    d1 = np.cross((p1 - p3), (p4 - p3))
    d2 = np.cross((p2 - p3), (p4 - p3))
    d3 = np.cross((p3 - p1), (p2 - p1))
    d4 = np.cross((p4 - p1), (p2 - p1))
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    else:
        return False


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