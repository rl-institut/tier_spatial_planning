import numpy as np
import pandas as pd

from fastapi_app.python import config

"""

This module is dedicated to generating demand time series for different types of consumers based on node data and 
various demand parameters. It involves calculating the number of households and enterprises, obtaining geographical 
coordinates, and selecting demand profiles. The module reads profile data, combines household and enterprise profiles, 
and calibrates the total profile based on user-specified calibration targets, either in kilowatts or kilowatt-hours. 
This calibrated profile is then used in the application to represent the demand time series for a given set of nodes 
and consumer types.
Note: This module was not developed by the TU Berlin but rather by the Rainer Limone Institute.
"""


def get_demand_time_series(nodes, demand_par_dict, all_profiles=None, df_only=True):
    num_households = len(nodes[(nodes['consumer_type'] == 'household') & (nodes['is_connected'] == True)].index)
    calibration_target_value, calibration_option = get_calibration_target(demand_par_dict)
    if all_profiles is None:
        all_profiles = pd.read_parquet(path=config.FULL_PATH_PROFILES, engine="pyarrow")
    df_hh_profile = combine_hh_profiles(all_profiles,
                                        num_households=num_households,
                                        demand_par_dict=demand_par_dict)
    enterprise_nodes = nodes[(nodes['consumer_type'] == 'enterprise') & (nodes['is_connected'] == True)]
    public_service_nodes = nodes[(nodes['consumer_type'] == 'public_service') & (nodes['is_connected'] == True)]
    df_ent_profile = combine_ent_or_pubs_profiles(all_profiles, enterprise_nodes)
    df_pub_profile = combine_ent_or_pubs_profiles(all_profiles, public_service_nodes)
    df, calibration_factor = calibrate_profiles(df_hh_profile,
                                                df_ent_profile,
                                                df_pub_profile,
                                                calibration_target_value,
                                                calibration_option)
    if df_only:
        return df / 1000
    else:
        return df / 1000, calibration_target_value, calibration_option, calibration_factor


def get_calibration_target(demand_par_dict):
    if demand_par_dict['maximum_peak_load'] is not None:
        value = float(demand_par_dict['maximum_peak_load'])
        calibration_option = 'kW'
    elif demand_par_dict['average_daily_energy'] is not None:
        value = float(demand_par_dict['average_daily_energy'])
        calibration_option = 'kWh'
    else:
        value = 1
        calibration_option = None
    return value, calibration_option


def combine_ent_or_pubs_profiles(all_profiles, enterprises):
    if enterprises is None or enterprises.empty:
        return pd.DataFrame()
    standard_ents = enterprises.query("consumer_type == 'enterprise'")
    common_ent_profile = all_profiles[
        "Enterprise_Large Load_Milling Machine"].copy()  # placeholder copy to keep same format before building up profile
    common_ent_profile *= 0
    if not standard_ents.empty:
        for enterprise_index in standard_ents.index:
            enterprise_type = standard_ents.loc[enterprise_index].consumer_detail.strip()
            column_select_string = "Enterprise_" + enterprise_type
            common_ent_profile += all_profiles[column_select_string]
    public_services = enterprises.query("consumer_type == 'public_service'")
    public_services_profile \
        = all_profiles["Enterprise_Large Load_Milling Machine"].copy()  # placeholder copy to keep same format before building up profile
    public_services_profile *= 0
    if not public_services.empty:
        for public_service_index in public_services.index:
            public_service_type = public_services.loc[public_service_index].consumer_detail.strip()
            column_select_string = "Public Service_" + public_service_type
            public_services_profile += all_profiles[column_select_string]
    large_load_ents = enterprises.query("(custom_specification.notnull()) & (consumer_type == 'enterprise')",
                                        engine='python')
    large_load_profile = all_profiles[
        "Enterprise_Large Load_Milling Machine"].copy()  # placeholder copy to keep same format before building up profile
    large_load_profile *= 0
    if not large_load_ents.empty:
        for enterprise_index in large_load_ents.index:
            large_loads_list = large_load_ents.loc[enterprise_index].custom_specification.split(';')
            # print("large_loads_list:", large_loads_list)
            if large_loads_list[0] != '':
                for load_type_and_count in large_loads_list:
                    load_count = int(load_type_and_count.split("x")[0].strip())
                    load_type = load_type_and_count.split("x")[1].split("(")[0].strip()
                    enterprise_type = large_load_ents.loc[enterprise_index].consumer_detail.strip()
                    column_select_string = "Enterprise_Large Load_" + load_type
                    large_load_profile += (load_count * all_profiles[column_select_string])
    total_non_household_profile = common_ent_profile + public_services_profile + large_load_profile
    return total_non_household_profile


def combine_hh_profiles(all_profiles, num_households, demand_par_dict):
    df_hh_profiles = \
        all_profiles["Household_Distribution_Based_Very Low Consumption"] * float(demand_par_dict["custom_share_1"]) + \
        all_profiles["Household_Distribution_Based_Low Consumption"] * float(demand_par_dict["custom_share_2"]) + \
        all_profiles["Household_Distribution_Based_Middle Consumption"] * float(demand_par_dict["custom_share_3"]) + \
        all_profiles["Household_Distribution_Based_High Consumption"] * float(demand_par_dict["custom_share_4"]) + \
        all_profiles["Household_Distribution_Based_Very High Consumption"] * float(demand_par_dict["custom_share_5"])
    df_hh_profiles *= num_households / 100
    return df_hh_profiles


def calibrate_profiles(df_hh_profile, df_ent_profile, df_pub_profile, calibration_target_value, calibration_option=None):
    calibration_factor = 1
    df_lst = [df_hh_profile, df_ent_profile, df_pub_profile]
    ts = [df for df in df_lst if not df.empty][0].index
    for i, df in enumerate(df_lst):
        if df.empty:
            df_lst[i] = pd.DataFrame(0, index=ts, columns=['value'])
    if calibration_option is not None:
        if calibration_option == "kWh":
            calibration_factor = calibration_target_value / ((df_hh_profile + df_ent_profile + df_pub_profile).sum() / 1000)
        elif calibration_option == "kW":
            calibration_factor = calibration_target_value / ((df_hh_profile + df_ent_profile + df_pub_profile).max() / 1000)
        for i, df in enumerate(df_lst):
            df_lst[i] = df_lst[i] * calibration_factor
    df = pd.concat(df_lst, axis=1)
    df.columns = ['households', 'enterprises', 'public_services']
    return df, calibration_factor


def default_wealth_share():
    wealth_share_dict = {'custom_share_1': 66.3,
                        'custom_share_2': 21.5,
                        'custom_share_3': 7.6,
                        'custom_share_4': 3.1,
                        'custom_share_5': 1.5}
    return wealth_share_dict


def demand_time_series_df():
    df = pd.DataFrame({'y': np.array([13.49953974, 15.83398798, 16.89947568, 18.20875497,
                                      23.60429479, 37.54596197, 80.07917413, 142.83629643,
                                      172.45226107, 141.46251121, 90.87287532, 70.77212158,
                                      68.94987379, 75.71314905, 83.42311357, 95.53603836,
                                      116.1458908, 152.12550107, 204.11813009, 217.96423835,
                                      175.92149604, 110.3719765, 48.28153063, 21.47630212]),
                       'Very High Consumption': np.array([13.77041436, 15.81503687, 16.63244513, 18.13944986,
                                                          26.64174449, 48.56398181, 124.6267382, 227.91122555,
                                                          273.32850081, 211.52258186, 132.97391816, 103.81524257,
                                                          105.93384848, 121.78107047, 135.28809975, 150.92480895,
                                                          181.3424163, 242.85534028, 331.34083495, 357.04599679,
                                                          294.07688899, 184.97017355, 72.60692784, 26.80880419]),
                       'High Consumption': np.array([14.29039556, 16.45040194, 17.68176054, 19.02752771,
                                                     24.90291366, 42.51510708, 96.50340108, 178.1295307,
                                                     211.25218502, 173.66860695, 114.51278482, 96.48181752,
                                                     100.68207069, 111.94970519, 124.78166031, 140.57781483,
                                                     163.82347078, 206.58609928, 272.7864012, 291.12552061,
                                                     240.92495333, 154.72806091, 63.91678491, 25.99007305]),
                       'Middle Consumption': np.array([13.49953974, 15.83398798, 16.89947568, 18.20875497,
                                                       23.60429479, 37.54596197, 80.07917413, 142.83629643,
                                                       172.45226107, 141.46251121, 90.87287532, 70.77212158,
                                                       68.94987379, 75.71314905, 83.42311357, 95.53603836,
                                                       116.1458908, 152.12550107, 204.11813009, 217.96423835,
                                                       175.92149604, 110.3719765, 48.28153063, 21.47630212]),
                       'Low Consumption': np.array([11.62191394, 13.46806407, 14.28451082, 15.81374831,
                                                    19.68194483, 31.48205428, 64.08022615, 110.92588889,
                                                    132.53335188, 105.77681489, 64.38242885, 45.08046125,
                                                    40.56064195, 41.63403794, 45.31661133, 52.97550959,
                                                    69.04889757, 97.24583476, 138.38286171, 151.72311467,
                                                    123.55244896, 77.98764392, 33.66924494, 15.96513716]),
                       'Very Low Consumption': np.array([4.07929076, 4.68156552, 5.14785365, 5.47445903, 7.87408584,
                                                         14.50133524, 27.36639231, 47.52312993, 57.04190414,
                                                         45.89645712,
                                                         26.67302567, 16.95623011, 15.12360097, 15.80745485,
                                                         18.06409324,
                                                         23.04509387, 31.45093103, 43.91518124, 58.83069314,
                                                         63.07259883,
                                                         50.77845948, 31.446005, 13.72102977, 5.95384269]),
                       'National': np.array([6.878804768980001, 7.950082814479999, 8.566178201619998, 9.27533270817, 12.417679945219998,
                                             21.282948090669997, 42.86819046414, 75.15314361857, 90.06857018810999, 72.47908266722001,
                                             43.9772818481, 30.86112595343, 28.697827788989997, 30.483016968389997, 33.95727485125,
                                             40.55115510699999, 52.32323178992, 71.63215689872, 97.69653366776, 105.3834658837,
                                             85.47975574876, 53.575437463259995, 23.0758509773, 10.21992548139]
                                            ),
                       'x': np.array(
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])})
    return df
