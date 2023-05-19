import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
import pandas as pd
import io


def df_to_xlsx(input_df, energy_flow_df, results_df, nodes_df, links_df):
    input_df = input_df.T
    input_df.columns = ["User specified input parameters"]
    input_df['Unit'] = ''
    input_df.index.name = ""
    input_df = input_df.T[['project_name',
    'project_description',
    'created_at',
    'updated_at',
    'start_date',
    'n_days',
    'temporal_resolution',
    'interest_rate',
    'project_lifetime',
    'connection_cable_capex',
    'connection_cable_lifetime',
    'connection_cable_max_length',
    'distribution_cable_capex',
    'distribution_cable_lifetime',
    'distribution_cable_max_length',
    'mg_connection_cost',
    'pole_capex',
    'pole_lifetime',
    'pole_max_n_connections',
    'allow_shs',
    'shs_lifetime',
    'shs_tier_five_capex',
    'shs_tier_four_capex',
    'shs_tier_one_capex',
    'shs_tier_three_capex',
    'shs_tier_two_capex'
    ]].T.fillna('')
    input_df.loc['project_lifetime', 'unit'] = 'years'
    input_df.loc['n_days', 'unit'] = 'days'
    input_df.loc['temporal_resolution', 'unit'] ='hours'
    input_df.loc['interest_rate', 'unit'] = '%'
    input_df.loc[['distribution_cable_capex', 'pole_capex', 'connection_cable_capex'], 'unit'] = 'USD/m'
    input_df.loc[['distribution_cable_lifetime', 'pole_lifetime', 'connection_cable_lifetime', 'project_lifetime',
                  'shs_lifetime'], 'unit'] = 'years'
    input_df.loc[['distribution_cable_max_length', 'pole_lifetime', 'connection_cable_max_length'], 'unit'] = 'm'
    input_df.loc[['shs_tier_one_capex', 'shs_tier_two_capex', 'shs_tier_three_capex', 'mg_connection_cost'], 'unit'] = 'USD'
    input_df.loc[['shs_tier_four_capex', 'shs_tier_five_capex'], 'unit'] = 'c/kWh'
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
    results_df.loc[['Average length connection cable', 'Average length distribution cable', 'Length connection cable',
                    'Length distribution cable'], 'Unit'] = 'm'
    results_df.loc[['CO2 savings'], 'Unit'] = 'tons'
    results_df.loc[['Cost fuel', 'Cost grid', 'Cost non renewable assets', 'Cost grid', 'Cost renewable assets',
                    'Cost shs'], 'Unit'] = 'USD'
    results_df.loc[['Diesel genset capacity', 'Inverter capacity', 'PV capacity', 'Rectifier capacity',
                    'Battery capacity', 'Surplus'], 'Unit'] = 'kWh'
    results_df = results_df.drop(['Time', 'Time grid design', 'Time energy system design', 'Battery to DC bus',
                                  'PV to DC bus', 'Rectifier to DC bus', 'Dc bus to battery', 'Dc bus to inverter',
                                  'Dc bus to surplus', 'Diesel genset to demand', 'Diesel genset to rectifier',
                                  'Fuel to diesel genset', 'Inverter to demand'])
    results_df.loc[['LCOE'], 'Unit'] = 'c/kWh'
    results_df = results_df.T
    results_df = results_df[['LCOE', 'Cost fuel', 'Cost grid', 'Cost non renewable assets', 'Cost grid',
                             'Cost renewable assets', 'Cost shs',
                             'PV capacity', 'Diesel genset capacity', 'Inverter capacity', 'Rectifier capacity',
                             'Battery capacity',
                             'Surplus', 'Surplus rate', 'Shortage total', 'Max voltage drop', 'RES share',
                             'Average length connection cable', 'Average length distribution cable',
                             'Length connection cable',
                             'Length distribution cable',
                             'N connection links', 'N consumers', 'N distribution links', 'N poles', 'N shs consumers',
                             'RES share', 'Surplus rate', 'Shortage total', 'Max voltage drop'
                             ]]
    results_df = results_df.T.reset_index()

    nodes_df = nodes_df.drop(columns=['surface_area', 'distribution_cost'])
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
                                                 energy_flow_df,
                                                 workbook.add_format({'align': 'right'}))
        sheet5 = 'links'
        links_df.to_excel(writer, sheet_name=sheet5, index=False)
        writer.sheets[sheet5] = set_column_width(writer.sheets[sheet5],
                                                 energy_flow_df,
                                                 workbook.add_format({'align': 'right'}))
        writer.save()
    xlsx_data = excel_file.getvalue()
    return io.BytesIO(xlsx_data)


def set_column_width(worksheet, df, format=None):
    for i, col in enumerate(df.columns):
        column_len = df[col].astype(str).str.len().max()
        column_len = max(column_len, len(col)) + 2
        if format:
            worksheet.set_column(i, i, column_len, format)
        else:
            worksheet.set_column(i, i, column_len)
    return worksheet


def format_first_col(df):
    df.iloc[:, 0] = df.iloc[:, 0].astype(str)\
        .str.replace('shs', 'SHS')\
        .str.replace('_', ' ')\
        .str.capitalize()\
        .str.replace('Mg', 'Mini-grid')\
        .str.replace('Lcoe', 'LCOE') \
        .str.replace('Pv', 'PV') \
        .str.replace(' dc ', ' DC ') \
        .str.replace('Co2', 'CO2') \
        .str.replace('Res', 'RES share')
    return df

def format_column_names(df):
    df.columns = [col.replace('_', ' ').capitalize() for col in df.columns]
    return df