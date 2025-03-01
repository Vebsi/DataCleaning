
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel files
files = ["C:/Users/vertt/Desktop/gardu2/data1.xlsx", 
         "C:/Users/vertt/Desktop/gardu2/data2.xlsx", 
         "C:/Users/vertt/Desktop/gardu2/data3.xlsx"]

dataframes = [pd.read_excel(file, sheet_name="Results") for file in files]

# Standardize column names
for df in dataframes:
    df.rename(columns={"Company name Latin alphabet": "Company Name"}, inplace=True)

# Merge all data
merged_df = pd.concat(dataframes, ignore_index=True)

# Replace missing values
merged_df.replace(["n.a.", "n.s."], 0, inplace=True)

# Transform data into panel format
def transform_to_panel(df, id_vars):
    value_columns = [col for col in df.columns if any(str(year) in col for year in range(1900, 2100))]
    melted = df.melt(id_vars=id_vars, value_vars=value_columns, var_name='Feature', value_name='Value')
    melted['Year'] = melted['Feature'].str.extract(r'(\d{4})').astype(int)
    melted['Feature'] = melted['Feature'].str.replace(r'\s*\d{4}$', '', regex=True).str.strip()
    return melted.pivot_table(index=id_vars + ['Year'], columns='Feature', values='Value', aggfunc='first').reset_index()

id_vars = ['Company Name', 'Country ISO code', 'BvD sectors']
panel_data = transform_to_panel(merged_df, id_vars)

# Rename columns
column_rename_map = {
    'Company Name': 'CN', 'Country ISO code': 'N', 'BvD sectors': 'S', 'Year': 'YR',
    'Accounts receivable': 'AR', 'Additions to Fixed Assets': 'AFA', 'Cost of goods sold': 'COGS',
    'Current ratio': 'CR', 'Deferred revenue': 'DR', 'Increase/Decrease in Accounts Payable': 'IDAP',
    'Net Cash from Operating Activities': 'CFO', 'Operating revenue (Turnover)': 'OR',
    'P/L for period [=Net income]': 'NI', 'Shareholders funds': 'SF', 'Total assets': 'TA',
    'Total liabilities': 'TL', 'Free Cash Flow': 'FCF'
}
panel_data.rename(columns=column_rename_map, inplace=True)

# Merge Interest Rate Data
interest_rate_data = {
    'Year': [2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005],
    '3_month': [-0.383, -0.309, -0.329, -0.319, -0.131, 0.078, 0.287, 0.187, 1.356, 1.006, 0.7, 2.892, 4.684, 3.725, 2.488],
    '12_month': [-0.249, -0.117, -0.186, -0.082, 0.06, 0.325, 0.556, 0.542, 1.947, 1.507, 1.248, 3.049, 4.745, 4.028, 2.844]}

interest_rate_df = pd.DataFrame(interest_rate_data)
panel_data = pd.merge(panel_data, interest_rate_df, left_on='YR', right_on='Year', how='left').drop(columns=['Year'])

# Create Lagged Interest Rate Variables
panel_data['3_month_l1'] = panel_data.groupby('CN')['3_month'].shift(1)
panel_data['12_month_l1'] = panel_data.groupby('CN')['12_month'].shift(1)
# Merge GDP Data
gdp_data = pd.read_excel("C:/Users/vertt/Desktop/gardu2/gdp.xlsx")
gdp_long = pd.melt(gdp_data, id_vars=["Country Code"], var_name="Year", value_name="GDP")
gdp_long['Country Code'] = gdp_long['Country Code'].str.strip()
panel_data = pd.merge(panel_data, gdp_long, left_on=['N', 'YR'], right_on=['Country Code', 'Year'], how='left').drop(columns=['Country Code', 'Year'])
panel_data['GDP_lagged'] = panel_data.groupby('N')['GDP'].shift(1)

# Remove rows with excessive missing values
panel_data = panel_data[(panel_data == 0).sum(axis=1) <= 3]


# Compute financial ratios
panel_data = panel_data[panel_data['SF'] != 0]
panel_data['ROE'] = panel_data['NI'] / panel_data['SF']
panel_data['ROA'] = panel_data['NI'] / panel_data['TA']
panel_data['FCF'] = panel_data['CFO'] - panel_data['AFA']

# Data Cleaning: Remove rows with zero values in key financials
panel_data = panel_data[(panel_data[['CFO', 'AFA', 'SF', 'ROE', 'ROA', 'FCF']] != 0).all(axis=1)]


# Rolling Mean Calculation for Total Assets (ATA)
panel_data = panel_data.sort_values(by=['CN', 'YR'])
panel_data['ATA'] = panel_data.groupby('CN')['TA'].rolling(2).mean().reset_index(level=0, drop=True)
panel_data.dropna(subset=['ATA'], inplace=True)

# Drop next-year columns if they exist
panel_data.drop(columns=['ROE_next_year', 'ROA_next_year', 'FCF_next_year', 'CFO_next_year'], errors='ignore', inplace=True)

# Create Lag Features
def create_lagged_column(df, column_name, group_by_column, lag=1):
    df = df.sort_values(by=[group_by_column, 'YR'])
    lagged_column_name = f"{column_name}_lag{lag}"
    df[lagged_column_name] = df.groupby(group_by_column)[column_name].shift(lag)
    return df

panel_data = create_lagged_column(panel_data, 'CFO', 'CN', lag=1)
panel_data = create_lagged_column(panel_data, 'FCF', 'CN', lag=1)
panel_data = create_lagged_column(panel_data, 'ROA', 'CN', lag=1)
panel_data = create_lagged_column(panel_data, 'ROE', 'CN', lag=1)

# Remove companies with excessive zero ROA, FCF, ROE values
company_roa_zero_count = panel_data[panel_data['ROA'] == 0].groupby('CN')['ROA'].count()
panel_data = panel_data[~panel_data['CN'].isin(company_roa_zero_count[company_roa_zero_count > 2].index)]

company_fcf_zero_count = panel_data[panel_data['FCF'] == 0].groupby('CN')['FCF'].count()
panel_data = panel_data[~panel_data['CN'].isin(company_fcf_zero_count[company_fcf_zero_count > 2].index)]

company_roe_zero_count = panel_data[panel_data['ROE'] == 0].groupby('CN')['ROE'].count()
panel_data = panel_data[~panel_data['CN'].isin(company_roe_zero_count[company_roe_zero_count > 2].index)]

# Create binary target variables
panel_data['Y_ROE'] = (panel_data['ROE_lag1'] > panel_data['ROE']).astype(int)
panel_data['Y_ROA'] = (panel_data['ROA_lag1'] > panel_data['ROA']).astype(int)
panel_data['Y_FCF'] = (panel_data['FCF_lag1'] > panel_data['FCF']).astype(int)
panel_data['Y_CFO'] = (panel_data['CFO_lag1'] > panel_data['CFO']).astype(int)

# Convert categorical columns to numeric codes
panel_data['N_coded'] = panel_data['N'].astype('category').cat.codes
panel_data['S_coded'] = panel_data['S'].astype('category').cat.codes

# Display distributions of Y values
print(panel_data[['Y_ROE', 'Y_ROA', 'Y_FCF', 'Y_CFO']].apply(pd.Series.value_counts))
print(panel_data[['Y_ROE', 'Y_ROA', 'Y_FCF', 'Y_CFO']].apply(lambda x: x.value_counts(normalize=True) * 100))

# Remove companies with insufficient data
company_counts = panel_data['CN'].value_counts()
panel_data = panel_data[panel_data['CN'].isin(company_counts[company_counts > 10].index)]

# Drop years after 2019
panel_data = panel_data[panel_data['YR'] <= 2019]


panel_data = panel_data.dropna()


# Save cleaned data
panel_data.to_excel("C:/Users/vertt/Desktop/CleanData.xlsx", index=False)
# %%
