
import datetime
import time


def convert_json_to_polygones_geojson(json_dict):
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
