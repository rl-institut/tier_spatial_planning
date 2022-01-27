import pandas as pd
import numpy as np
import os


def create_empty_nodes_df():
    """ 
    Creates an empty DataFrame for representing nodes.

    Output
    ------
    (pandas.DataFrame): DataFrame representing nodes.
     """
    return pd.DataFrame(
        {
            'label':
            pd.Series([], dtype=str),
            'latitude':
            pd.Series([], dtype=np.dtype(float)),
            'longitude':
            pd.Series([], dtype=np.dtype(float)),
            'area':
            pd.Series([], dtype=np.dtype(float)),
            'node_type':
            pd.Series([], dtype=np.dtype(str)),
            'customer_type':
            pd.Series([], dtype=np.dtype(str)),
            'peak_demand':
            pd.Series([], dtype=np.dtype(float)),
            'demand_type':
            pd.Series([], dtype=np.dtype(str)),
            'is_connected':
            pd.Series([], dtype=np.dtype(bool)),
            'how_added':
            pd.Series([], dtype=np.dtype(str))
        }
    ).set_index('label')


def create_empty_links_df():
    return pd.DataFrame(
        {
            'label':
            pd.Series([], dtype=str),
            'latitude_from':
            pd.Series([], dtype=np.dtype(float)),
            'longitude_from':
            pd.Series([], dtype=np.dtype(float)),
            'latitude_to':
            pd.Series([], dtype=np.dtype(float)),
            'longitude_to':
            pd.Series([], dtype=np.dtype(float)),
            'type':
            pd.Series([], dtype=np.dtype(str)),
            'distance':
            pd.Series([], dtype=np.dtype(float)),
        }
    ).set_index('label')


# taken from sgdot (temporary comment)


def make_folder(folder):
    """
    If no folder of the given name already exists, create new one.

    Parameters
    ----------
    folder: str
        Name of the folder to be created.
    """

    if not os.path.exists(folder):
        parent_folders = folder.split('/')
        for i in range(1, len(parent_folders) + 1):
            path = ''
            for x in parent_folders[0:i]:
                path += x + '/'
            if not os.path.exists(path[0:-1]):
                os.mkdir(path[0:-1])
