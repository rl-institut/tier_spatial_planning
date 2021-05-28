import numpy as np
import datetime
import time

from shapely.geometry.base import geometry_type_name

import fastapi_app.tools.convertion as conv
from shapely import geometry


def convert_json_to_polygones_geojson(json_dict):
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


def get_dict_with_mean_coordinate_from_geojson(geojson: dict):
    """
    This function creates a dictionnary with the @id of each building as a key
    and the mean loaction of the building as value in the form [lat, long].

    Parameters
    ----------
        geojson (dict):
            Dictionary containing the geojson data of the building of a
            specific area. Output of the
            tools.conversion.convert_json_to_polygones_geojson function.

    Returns
    -------
        (1)
        Dict containing the @id of each building as a key
        and the mean loaction of the building as value in the form [long, lat].

        (2)
        Dict containing the @id of each building as a key
        and the surface area of the buildings.
    """

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
                xy_coordinates.append(conv.xy_coordinates_from_latitude_longitude(
                    latitude=latitudes_longitudes[edge][0],
                    longitude=latitudes_longitudes[edge][1],
                    ref_latitude=reference_coordinate[0],
                    ref_longitude=reference_coordinate[1]))
            surface_area = geometry.Polygon(xy_coordinates).area
            building_mean_coordinates[building["property"]["@id"]] = mean_coord
            building_surface_areas[building["property"]["@id"]] = surface_area

    return building_mean_coordinates, building_surface_areas


def are_segment_crossing(segment1, segment2):
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
    """ Function that checks whether or not 2D point lies within boundaries

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
