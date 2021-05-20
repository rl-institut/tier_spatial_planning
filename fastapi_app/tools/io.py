import pandas as pd


def create_empty_nodes_df():
    """ Creates an empty DataFrame for representing nodes.

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
            'node_type':
            pd.Series([], dtype=np.dtype(str)),
            'type_fixed':
            pd.Series([], dtype=np.dtype(bool)),
            'required_capacity':
            pd.Series([], dtype=np.dtype(float)),
            'max_power':
            pd.Series([], dtype=np.dtype(float))
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
