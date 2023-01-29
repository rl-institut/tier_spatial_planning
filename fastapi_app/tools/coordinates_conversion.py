import math


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

    Output
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


def latitude_longitude_from_xy_coordinates(x_coord, y_coord, ref_latitude, ref_longitude):
    """ This function converts (x, y) plane coordinates into
    (latitude, longitude) coordinates using reference latitude
    and longitude.

    Parameters
    ----------
        x_coord (float):
            x coordinate.

        y_coord (float):
            y coordinate.

        ref_latitude (float):
            Reference latitude in degree.

        ref_longitude (float):
            Reference longitude in degree.

    Output
    ------
        (tuple):
            (latitude, longitude) coordinates in degree. 
    """
    r = 6371000     # Radius of the earth [m]

    longitude = ref_longitude + math.degrees(x_coord / (r * math.cos(ref_latitude)))

    latitude = ref_latitude + math.degrees(y_coord / r)

    return latitude, longitude
