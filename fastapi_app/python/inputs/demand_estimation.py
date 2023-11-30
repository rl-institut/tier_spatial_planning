import pandas as pd

from fastapi_app import config


def get_demand_time_series(nodes, demand_par_dict, all_profiles=None, distribution_lookup=None):
    # print(nodes)

    num_households = get_number_of_households(nodes)
    lat, lon = get_location_gps(nodes).values()
    hh_demand_option = get_user_household_demand_option_selection(demand_par_dict)
    calibration_target_value, calibration_option = get_calibration_target(demand_par_dict)
    if all_profiles is None:
        all_profiles = read_all_profiles(config.FULL_PATH_PROFILES)
    if distribution_lookup is None:
        distribution_lookup = read_distribution_lookup(config.FULL_PATH_DISTRIBUTIONS)
    df_hh_profiles = combine_hh_profiles(all_profiles,
                                         lat=lat,
                                         lon=lon,
                                         num_households=num_households,
                                         distribution_lookup=distribution_lookup,
                                         demand_par_dict=demand_par_dict,
                                         option=hh_demand_option)
    enterprises = get_all_enterprise_customer_nodes(nodes)
    # print(enterprises)
    df_ent_profiles = combine_ent_profiles(all_profiles, enterprises)

    # print("demand_par_dict:", demand_par_dict)
    # print("use_custom_shares", demand_par_dict["use_custom_shares"])
    # print("enterprises consumer type:", enterprises.consumer_type.iloc[0])
    # print("enterprises consumer detail:", enterprises.consumer_detail.iloc[0])
    # print("enterprises node_type:", enterprises.node_type.iloc[0])
    # print("enterprises data:", enterprises.custom_specification.iloc[0])
    # print("hh_demand_option: ", hh_demand_option)
    # print("calibration_option:", calibration_option)
    # print("calibration_target_value:", calibration_target_value)

    calibrated_profile = combine_and_calibrate_total_profile(
        df_hh_profiles=df_hh_profiles,
        df_ent_profiles=df_ent_profiles,
        calibration_target_value=calibration_target_value,
        calibration_option=calibration_option) / 1000
    # calibration totals/setpoints are in kW
    # profiles are still in W
    # print("calibrated_profile_max:", calibrated_profile.max())
    # print("calibrated_profile_sum:", calibrated_profile.sum())

    return calibrated_profile


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


def get_number_of_households(nodes):
    num_households = len(nodes[(nodes['consumer_type'] == 'household') &
                               (nodes['is_connected'] == True)].index)
    return num_households


def get_number_of_enterprise(nodes):
    num_enterprise = len(nodes[((nodes['consumer_type'] == 'enterprise') |
                                (nodes['consumer_type'] == 'public_service')) &
                               (nodes['is_connected'] == True)].index)
    return num_enterprise


def get_location_gps(nodes):
    lat = nodes["latitude"].median()
    lon = nodes["longitude"].median()
    return {"lat": lat, "lon": lon}


def get_user_household_demand_option_selection(demand_par_dict):
    # Dummy function
    # Gives chosen selection option from radio button list
    option = demand_par_dict['household_option']
    return option


def get_all_enterprise_customer_nodes(nodes):
    # Dummy function
    # Returns a list or dataframe of all the enterprise node "strings" of the community
    nodes = nodes[((nodes['consumer_type'] == 'enterprise') |
                   (nodes['consumer_type'] == 'public_service')) &
                  (nodes['is_connected'] == True)]
    return nodes


def read_all_profiles(filepath):
    df_all_profiles = pd.read_parquet(path=filepath, engine="pyarrow")
    return df_all_profiles


def read_distribution_lookup(filepath):
    distribution_lookup = pd.read_parquet(path=filepath, engine="pyarrow")
    return distribution_lookup


def read_wealth_lookup(filepath=None):
    ##Dummy function

    # wealth_lookup = pd.read_parquet(path = filepath, engine = "pyarrow")
    wealth_lookup = 0
    return wealth_lookup


def resample_to_hourly(profile):
    # Not used yet. Full profile results uploaded and read already as hourly timeseries
    resampled_mean = profile.resample("H").mean()
    resampled_min = profile.resample("H").min()
    resampled_max = profile.resample("H").max()

    return resampled_mean, resampled_min, resampled_max


def combine_ent_profiles(all_profiles, enterprises):
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

    public_services_profile = all_profiles[
        "Enterprise_Large Load_Milling Machine"].copy()  # placeholder copy to keep same format before building up profile
    public_services_profile *= 0

    if not public_services.empty:
        for public_service_index in public_services.index:
            public_service_type = public_services.loc[public_service_index].consumer_detail.strip()
            column_select_string = "Public Service_" + public_service_type
            public_services_profile += all_profiles[column_select_string]

    large_load_ents = enterprises.query("(custom_specification.notnull()) & (consumer_type == 'enterprise')",
                                        engine='python')
    # print(large_load_ents.custom_specification)
    # print(large_load_ents)

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


def hh_location_estimate(all_profiles, lat, lon, num_households, wealth_lookup):
    # Dummy function - still tests h3.geo_to_h3() functionality
    default_location_estimate = "Household_Location_Estimate_Middle Wealth"
    # hh_consumption_estimate = wealth_lookup[H3_cell]

    df_hh_profiles = all_profiles[default_location_estimate] * num_households

    return df_hh_profiles


def combine_hh_profiles_distribution_based(all_profiles, num_households, percentages):
    hh_per_cat = (percentages * num_households).round(0)  # does not check for percentages adding up to 100%

    df_hh_profiles_distribution_based = \
        all_profiles["Household_Distribution_Based_Very Low Consumption"] * hh_per_cat["Very Low"] + \
        all_profiles["Household_Distribution_Based_Low Consumption"] * hh_per_cat["Low"] + \
        all_profiles["Household_Distribution_Based_Middle Consumption"] * hh_per_cat["Middle"] + \
        all_profiles["Household_Distribution_Based_High Consumption"] * hh_per_cat["High"] + \
        all_profiles["Household_Distribution_Based_Very High Consumption"] * hh_per_cat["Very High"]

    return df_hh_profiles_distribution_based


def combine_hh_profiles(all_profiles, lat, lon, num_households, distribution_lookup, demand_par_dict, option="Default"):
    if demand_par_dict["use_custom_shares"] == 0:
        if option == 0:
            percentages = distribution_lookup.loc["National"]["percentages"]
            df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

        elif option == 1:
            percentages = distribution_lookup.loc["South South"]["percentages"]
            df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

        elif option == 2:
            percentages = distribution_lookup.loc["North West"]["percentages"]
            df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

        elif option == 3:
            percentages = distribution_lookup.loc["North Central"]["percentages"]
            df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

        else:
            percentages = distribution_lookup.loc["National"]["percentages"]
            df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

    elif demand_par_dict["use_custom_shares"] == 1:
        percentages = distribution_lookup.loc["National"]["percentages"]  # placeholder to be overwritten below

        percentages["Very Low"] = float(demand_par_dict["custom_share_1"])
        percentages["Low"] = float(demand_par_dict["custom_share_2"])
        percentages["Middle"] = float(demand_par_dict["custom_share_3"])
        percentages["High"] = float(demand_par_dict["custom_share_4"])
        percentages["Very High"] = float(demand_par_dict["custom_share_5"])

        percentages = percentages / 100  # input percentages are 0-100, code percentages need to be 0-1. This does not normalize or check adding to 100.

        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

    else:  # fallback default in case "use_custom_shares" gets unexpected value.
        percentages = distribution_lookup.loc["National"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)

    return df_hh_profiles


def combine_and_calibrate_total_profile(df_hh_profiles, df_ent_profiles, calibration_target_value,
                                        calibration_option=None):
    if df_ent_profiles.empty:
        df_total_profile = df_hh_profiles
    elif df_hh_profiles.empty:
        df_total_profile = df_ent_profiles
    else:
        df_total_profile = df_hh_profiles + df_ent_profiles

    if calibration_option is not None:
        if calibration_option == "kWh":
            uncalibrated_profile_total = df_total_profile.sum() / 1000
            calibration_factor = calibration_target_value / uncalibrated_profile_total

            df_total_profile = df_total_profile * calibration_factor

        elif calibration_option == "kW":
            uncalibrated_profile_max = df_total_profile.max() / 1000
            calibration_factor = calibration_target_value / uncalibrated_profile_max

            df_total_profile = df_total_profile * calibration_factor

    return df_total_profile
