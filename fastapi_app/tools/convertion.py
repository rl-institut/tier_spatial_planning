
import datetime
import time


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

    node_coordinates = {element["id"]:  [element["lon"], element["lat"]]
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
