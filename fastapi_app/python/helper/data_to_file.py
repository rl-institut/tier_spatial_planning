import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
import os
import io
import numpy as np
import pandas as pd

"""
This module contains functions for generating an Excel file based on the results of the user-project. It includes 
functionality to format data from various DataFrame inputs like energy_system_design, energy_flow_df, and 
results_df into structured sheets within an Excel file. The module handles the formatting of data such as renaming 
columns, setting units, and adjusting column widths for readability. The Excel file is then created using the 
pandas.ExcelWriter class.
"""


def consumer_data_to_file(df, file_type):
    if df.empty:
        df = pd.DataFrame(columns=['latitude', 'longitude', 'consumer_type', 'custom_specification', 'shs_options', 'consumer_detail'])
    else:
        df = df.drop(columns=['is_connected', 'how_added', 'node_type'])
    if file_type == 'xlsx':
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='xlsxwriter')
        output.seek(0)
        return io.BytesIO(output.getvalue())
    elif file_type == 'csv':
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return io.StringIO(output.getvalue())


def check_imported_consumer_data(df):
    if df.empty:
        return None, 'No data could be read.'
    df.columns = [col.strip().lower() for col in df.columns]
    if 'latitude' not in df.columns:
        return None, 'Column with title \'latitude\' is missing.'
    if 'longitude' not in df.columns:
        return None, 'Column with title \'longitude\' is missing.'
    df['is_connected'] = True
    df['how_added'] = 'automatic'
    df['node_type'] = 'consumer'
    df = df.replace('', np.nan)
    df['consumer_detail'] = df['consumer_detail'].fillna('') if 'consumer_detail' in df.columns else ''
    df['consumer_type'] = df['consumer_type'].fillna('household') if 'consumer_type' in df.columns else 'household'
    df['custom_specification'] = df['custom_specification'].fillna('') if 'custom_specification' in df.columns else ''
    df['shs_options'] = df['shs_options'].fillna(0) if 'shs_options' in df.columns else 0
    df['shs_options'] = df['shs_options'].fillna(0)
    allowed_values = ['household', 'enterprise', 'public_service']
    falsy_values = set(df['consumer_type'].unique()) - set(allowed_values)
    if len(falsy_values) > 0:
        return None, f"Allowed values of column \'consumer_type\' are {allowed_values}. Falsy values passed: {list(falsy_values)}."
    allowed_values_shs_option = [0, 1]
    falsy_values_shs_option = set(df['shs_options'].unique()) - set(allowed_values_shs_option)
    if len(falsy_values_shs_option) > 0:
        return None, f"Allowed values of column 'shs_options' are {allowed_values_shs_option}. Falsy values passed: {list(falsy_values_shs_option)}."
    valid_consumer_details = ['Food_Groceries', 'Food_Restaurant', 'Food_Bar', 'Food_Drinks',
                              'Food_Fruits or vegetables', 'Trades_Tailoring', 'Trades_Beauty or Hair',
                              'Trades_Metalworks', 'Trades_Car or Motorbike Repair', 'Trades_Carpentry',
                              'Trades_Laundry', 'Trades_Cycle Repair', 'Trades_Shoemaking', 'Retail_Medical',
                              'Retail_Clothes and accessories', 'Retail_Electronics', 'Retail_Other',
                              'Retail_Agricultural', 'Digital_Mobile or Electronics Repair', 'Digital_Digital Other',
                              'Digital_Cybercafé', 'Digital_Cinema or Betting', 'Digital_Photostudio',
                              'Agricultural_Mill or Thresher or Grater', 'Agricultural_Other', '', 'default',
                              'Health_Health Centre', 'Health_Clinic', 'Health_CHPS', 'Education_School', 'Education_School_noICT']

    valid_custom_specifications = ['Milling Machine (7.5kW)', 'Crop Dryer (8kW)', 'Thresher (8kW)',
                                   'Grinder (5.2kW)', 'Sawmill (2.25kW)', 'Circular Wood Saw (1.5kW)',
                                   'Jigsaw (0.4kW)', 'Drill (0.4kW)', 'Welder (5.25kW)', 'Angle Grinder (2kW)', '']
    falsy_values_consumer_detail = set(df['consumer_detail'].unique()) - set(valid_consumer_details)
    if len(falsy_values_consumer_detail) > 0:
        return None, f"Allowed values of column 'consumer_detail' are {valid_consumer_details}. Falsy values passed: {list(falsy_values_consumer_detail)}."

    custom_loads = df[df['custom_specification'] != '']['custom_specification'].unique()
    processed_loads = []
    non_matching_values = []
    for load in custom_loads:
        if load[0].isdigit() and ' x ' in load:
            processed_loads.append(load.split(' x ', 1)[1])
        else:
            non_matching_values.append(load)
    if len(non_matching_values) > 0:
        return None, f"Values of 'custom_specification' must start with an integer followed by \" x \"  {valid_custom_specifications}. Falsy values passed: {list(non_matching_values)}."
    falsy_values_custom_specification = set(processed_loads) - set(valid_custom_specifications)
    if len(falsy_values_custom_specification) > 0:
        return None, f"Allowed values of column 'custom_specification' are {valid_custom_specifications}. Falsy values passed: {list(falsy_values_custom_specification)}."
    columns_types = {'latitude': float, 'longitude': float, 'shs_options': int, 'consumer_type': str, 'custom_specification': str,
                     'is_connected': bool}
    for column, dtype in columns_types.items():
        try:
            df[column] = df[column].astype(dtype)
        except ValueError as e:
            return None, f"Error converting '{column}' to {dtype.__name__}: {str(e)}"
    if df['latitude'].max() - df['latitude'].min() > float(os.environ.get("MAX_LAT_LON_DIST", 0.15)) \
            or df['longitude'].max() - df['longitude'].min() > float(os.environ.get("MAX_LAT_LON_DIST", 0.15)):
        return None, f"Distance between consumers exceeds maximum allowed distance."
    nigeria_bounds = {'latitude_min': 4.2, 'latitude_max': 13.9, 'longitude_min': 2.7, 'longitude_max': 14.7}
    out_of_bounds_latitudes = df[(df['latitude'] < nigeria_bounds['latitude_min']) | (df['latitude'] > nigeria_bounds['latitude_max'])]
    out_of_bounds_longitudes = df[(df['longitude'] < nigeria_bounds['longitude_min']) | (df['longitude'] > nigeria_bounds['longitude_max'])]
    if not out_of_bounds_latitudes.empty or not out_of_bounds_longitudes.empty:
        return None, (f"Error: Some latitude/longitude values are outside the bounds of Nigeria.\n"
                      f"Latitude must be between {nigeria_bounds['latitude_min']} and {nigeria_bounds['latitude_max']}.\n"
                      f"Longitude must be between {nigeria_bounds['longitude_min']} and {nigeria_bounds['longitude_max']}.")
    df = df[['latitude', 'longitude', 'how_added', 'node_type', 'consumer_type', 'custom_specification', 'shs_options', 'consumer_detail',
             'is_connected']]
    return df, ''


def project_data_df_to_xlsx(input_df, energy_system_design, energy_flow_df, results_df, nodes_df, links_df):
    input_df = pd.concat([input_df.T, energy_system_design.T])
    input_df.columns = ["User specified input parameters"]
    input_df.index.name = ""
    input_df = input_df.rename(index={'shs_max_grid_cost': 'shs_max_specific_marginal_grid_cost'})
    input_df['Unit'] = ''
    input_df = input_df.drop(['status', 'temporal_resolution'])
    input_df.index.str.replace('__parameters__', '_parameter: ')
    input_df.index.str.replace('__settings__', '_settings: ')
    input_df.loc['n_days', 'Unit'] = 'days'
    input_df.loc['interest_rate', 'Unit'] = '%'
    input_df.loc[['distribution_cable_capex', 'pole_capex', 'connection_cable_capex'], 'Unit'] = 'USD/m'
    input_df.loc[['distribution_cable_lifetime', 'pole_lifetime', 'connection_cable_lifetime', 'project_lifetime',
                  ], 'Unit'] = 'years'
    input_df.loc[input_df.index.str.contains('lifetime'), 'Unit'] = 'years'
    input_df.loc[input_df.index.str.contains('length'), 'Unit'] = 'm'
    input_df.loc[input_df.index.str.contains('__capex'), 'Unit'] = 'USD/kWh'
    input_df.loc[input_df.index.str.contains('__opex'), 'Unit'] = 'USD/(kW a)'
    input_df.loc[input_df.index.str.contains('__fuel'), 'Unit'] = 'USD/l'
    input_df.loc[input_df.index.str.contains('__fuel_cost'), 'Unit'] = 'USD/l'
    input_df.loc[input_df.index.str.contains('__fuel_lhv'), 'Unit'] = 'kWh/kg'
    input_df.loc[input_df.index.str.contains('_capacity'), 'Unit'] = 'kWh'
    input_df.loc[['battery__parameters__capex'], 'Unit'] = 'USD/kWh'
    input_df.loc[['mg_connection_cost'], 'Unit'] = 'USD'
    input_df.loc[['shs_max_specific_marginal_grid_cost'], 'Unit'] = 'c/kWh'
    input_df = input_df.reset_index()
    input_df = format_first_col(input_df)
    cols = [col.replace('_', ' ').capitalize() + ' [kW]'
            if 'content' not in col
            else col.replace('_', ' ').capitalize() + ' [kWh]' for col in energy_flow_df.columns]
    energy_flow_df.columns = cols
    energy_flow_df = energy_flow_df.reset_index()
    results_df = results_df.T.reset_index()
    results_df['Unit'] = ''
    results_df.columns = ['', 'Value', 'Unit']
    results_df = format_first_col(results_df)
    results_df = results_df.set_index('')
    results_df.loc[results_df.index.str.contains('ength'), 'Unit'] = 'm'
    results_df.loc[results_df.index.str.contains('CO2'), 'Unit'] = 't/a'
    results_df.loc[results_df.index.str.contains('Upfront'), 'Unit'] = 'USD'
    results_df.loc[results_df.index.str.contains('Cost'), 'Unit'] = 'USD/a'
    results_df.loc[results_df.index.str.contains('Epc'), 'Unit'] = 'USD/a'
    results_df.loc[results_df.index.str.contains('capacity'), 'Unit'] = 'USD/kW'
    results_df.loc[['Battery capacity'], 'Unit'] = 'USD/kWh'
    results_df.loc[['Max voltage drop', 'RES share', 'Surplus rate', 'Shortage total', 'Max shortage'], 'Unit'] = '%'
    results_df.loc[['Average annual demand per consumer', 'Fuel consumption', 'Total annual consumption', 'Surplus'],
    'Unit'] = 'kWh/a'
    results_df = results_df[~results_df.index.str.contains('Time')]
    results_df = results_df[~results_df.index.str.contains(' to ')]
    results_df = results_df.drop('Infeasible')
    results_df.loc[['LCOE'], 'Unit'] = 'c/kWh'
    results_df.loc[['Base load', 'Peak demand'], 'Unit'] = 'kW'
    results_df = results_df.T
    results_df = results_df.T.reset_index()

    nodes_df = nodes_df.drop(columns=['distribution_cost', 'parent'])
    nodes_df = format_column_names(nodes_df)
    links_df = links_df[['link_type', 'length', 'lat_from', 'lon_from', 'lat_to', 'lon_to']]
    links_df = format_column_names(links_df)
    excel_file = io.BytesIO()
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        sheet1 = 'results'
        results_df.to_excel(writer, sheet_name=sheet1, index=False)
        worksheet1 = writer.sheets[sheet1]
        format1 = workbook.add_format({'align': 'left'})
        format2 = workbook.add_format({'align': 'right'})
        col1_width = results_df.iloc[:, 0].astype(str).str.len().max()
        col2_width = results_df.iloc[:, 1].astype(str).str.len().max()
        col3_width = results_df.iloc[:, 2].astype(str).str.len().max()
        worksheet1.set_column(0, 0, col1_width, format1)
        worksheet1.set_column(1, 1, col2_width, format2)
        worksheet1.set_column(2, 2, col3_width, format1)
        sheet2 = 'power time series'
        energy_flow_df.to_excel(writer, sheet_name=sheet2, index=False)
        writer.sheets[sheet2] = set_column_width(writer.sheets[sheet2],
                                                 energy_flow_df,
                                                 workbook.add_format({'align': 'right'}))
        sheet3 = 'user specified input parameters'
        input_df.to_excel(writer, sheet_name=sheet3, index=False)
        worksheet3 = writer.sheets[sheet3]
        format1 = workbook.add_format({'align': 'left'})
        format2 = workbook.add_format({'align': 'right'})
        col1_width = input_df.iloc[:, 0].astype(str).str.len().max()
        col2_width = input_df.iloc[:, 1].astype(str).str.len().max()
        col3_width = input_df.iloc[:, 2].astype(str).str.len().max()
        worksheet3.set_column(0, 0, col1_width, format1)
        worksheet3.set_column(1, 1, col2_width, format2)
        worksheet3.set_column(2, 2, col3_width, format1)
        sheet4 = 'nodes'
        nodes_df.to_excel(writer, sheet_name=sheet4, index=False)
        writer.sheets[sheet4] = set_column_width(writer.sheets[sheet4],
                                                 nodes_df,
                                                 workbook.add_format({'align': 'right'}))
        sheet5 = 'links'
        links_df.to_excel(writer, sheet_name=sheet5, index=False)
        writer.sheets[sheet5] = set_column_width(writer.sheets[sheet5],
                                                 links_df,
                                                 workbook.add_format({'align': 'right'}))
    xlsx_data = excel_file.getvalue()
    return io.BytesIO(xlsx_data)


def set_column_width(worksheet, df, col_format=None):
    for i, col in enumerate(df.columns):
        column_len = df[col].astype(str).str.len().max()
        column_len = max(column_len, len(col)) + 2
        column_len = min(column_len, 150)
        if col_format:
            worksheet.set_column(i, i, column_len, col_format)
        else:
            worksheet.set_column(i, i, column_len)
    return worksheet


def format_first_col(df):
    df.iloc[:, 0] = df.iloc[:, 0].astype(str) \
        .str.replace('shs', 'SHS') \
        .str.replace('_', ' ') \
        .str.capitalize() \
        .str.replace('Mg', 'Mini-grid') \
        .str.replace('Lcoe', 'LCOE') \
        .str.replace('Pv', 'PV') \
        .str.replace(' dc ', ' DC ') \
        .str.replace('Co2', 'CO2') \
        .str.replace('Res', 'RES share')
    return df


def format_column_names(df):
    df.columns = [col.replace('_', ' ').capitalize() for col in df.columns]
    return df
