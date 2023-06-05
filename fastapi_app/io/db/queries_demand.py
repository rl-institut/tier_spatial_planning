import pandas as pd
import numpy as np
import h3
import plotly
import plotly.express as px
import plotly.figure_factory as ff

def get_location_GPS():
    #Dummy function
    #gives location of mean value, first value, or value of power house of project site
    lat = 0
    lon = 0
    return {"lat" : lat, "lon": lon}

def get_user_household_demand_option_selection():
    #Dummy function
    #Gives chosen selection option from radio button list
    option = "National"
    return option

def get_user_calibration_option_selection():
    #Dummy function
    #Gives chosen selection option from calibration radio button list
    option = "kW"
    return option

def get_all_enterprise_customer_nodes():
    #Dummy function
    #Returns a list or dataframe of all of the enterprise node "strings" of the community
    nodes = ["placeholder"] * 5
    
    return nodes

def read_all_profiles(filepath):
    df_all_profiles = pd.read_parquet(path = filepath, engine = "pyarrow")
    
    return df_all_profiles

def read_distribution_lookup(filepath):
    distribution_lookup = pd.read_parquet(path = filepath, engine = "pyarrow")
    return distribution_lookup

def read_wealth_lookup(filepath):
    ##Dummy function
    
    #wealth_lookup = pd.read_parquet(path = filepath, engine = "pyarrow")
    wealth_lookup = 0
    return wealth_lookup

def plot_profiles_hourly(profiles):
    ##TODO
    plotly_object = "placeholder"
    return plotly_object

def plot_profiles_1_minute(profiles):
    ##TODO
    plotly_object = "placeholder"
    return plotly_object

def resample_to_hourly(profile):
    ##TODO
    #Not used yet. Full profile results uploaded and read already as hourly timeseries
    resampled_mean = profile.resample("H").mean()
    resampled_min = profile.resample("H").min()
    resampled_max = profile.resample("H").max()
    
    return resampled_mean, resampled_min, resampled_max
    
def combine_ent_profiles(all_profiles, nodes):
    
    default_profile_name = "Enterprise_Food_Bar"
    
    ##regex or str.split to parse enterprise users
    ##compile total list of enterprises and heavy loads
    enterprise_nodes_list = nodes
    
    node_count = len(enterprise_nodes_list)
    
    default_ent_profile = all_profiles[default_profile_name]
    
    ##combine profiles
    if node_count == 0:
        zero_ent_profile = all_profiles[default_profile_name] * 0
        return zero_ent_profile
    elif node_count > 0:
        return default_ent_profile * node_count
    else:
        return default_ent_profile

def hh_location_estimate(all_profiles, lat, lon, num_households, wealth_lookup):
    
    #Dummy function - still tests h3.geo_to_h3() functionality
    default_location_estimate = "Household_Location_Estimate_Middle Wealth"
    H3_cell = h3.geo_to_h3(lat, lon, resolution = 8)
    
    #hh_consumption_estimate = wealth_lookup[H3_cell]
    
    df_hh_profiles = all_profiles[default_location_estimate] * num_households
    
    return df_hh_profiles
    
def combine_hh_profiles_distribution_based(all_profiles, num_households, percentages):
    
    #This code should work as expected
    hh_per_cat = (percentages * num_households).round(0)

    df_hh_profiles_distribution_based = \
    all_profiles["Household_Distribution_Based_Very Low Consumption"] * hh_per_cat["Very Low"] + \
    all_profiles["Household_Distribution_Based_Low Consumption"] * hh_per_cat["Low"] + \
    all_profiles["Household_Distribution_Based_Middle Consumption"] * hh_per_cat["Middle"] + \
    all_profiles["Household_Distribution_Based_High Consumption"] * hh_per_cat["High"] + \
    all_profiles["Household_Distribution_Based_Very High Consumption"] * hh_per_cat["Very High"]
     
    return df_hh_profiles_distribution_based 
    
def combine_hh_profiles(all_profiles, lat, lon, num_households, distribution_lookup, option = "Default"):
    
    default_profile_name = "Household_Distribution_Based_Very Low Consumption"
    
    if option == "National":
        percentages = distribution_lookup.loc["National"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    elif option == "South South":
        percentages = distribution_lookup.loc["South South"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    elif option == "North West":
        percentages = distribution_lookup.loc["North West"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    elif option == "North Central":
        percentages = distribution_lookup.loc["North Central"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    elif option == "Location Estimate":
        df_hh_profiles = hh_location_estimate(all_profiles, lat, lon, num_households, wealth_lookup)
    
    elif option == "Default":
        percentages = distribution_lookup.loc["National"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    else:
        percentages = distribution_lookup.loc["National"]["percentages"]
        df_hh_profiles = combine_hh_profiles_distribution_based(all_profiles, num_households, percentages)
        
    return df_hh_profiles

def combine_and_calibrate_total_profile(df_hh_profiles, df_ent_profiles, calibration_target_value, calibration_option = None):
    
    df_total_profile = df_ent_profiles + df_hh_profiles
    
    if calibration_option != None:
        if calibration_option == "kWh":
            ##TODO: calibrate to daily kWh total
            #This doesn't do anything yet, the version below does
            calibration_factor = 1
            df_total_profile = df_total_profile * calibration_factor
            
        elif calibration_option == "kW":
            uncalibrated_profile_max = df_total_profile.max() / 1000
            calibration_factor = calibration_target_value / uncalibrated_profile_max 
            
            df_total_profile = df_total_profile * calibration_factor
            
    return df_total_profile