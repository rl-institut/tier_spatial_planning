import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
import pandas as pd
import io

"""
This module contains functions for generating an Excel file based on the results of the user-project. It includes 
functionality to format data from various DataFrame inputs like energy_system_design, energy_flow_df, and 
results_df into structured sheets within an Excel file. The module handles the formatting of data such as renaming 
columns, setting units, and adjusting column widths for readability. The Excel file is then created using the 
pandas.ExcelWriter class.
"""

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
        writer.save()
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
