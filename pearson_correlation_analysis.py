import os
import pandas as pd

# Specify the relative folder path
folder_path = 'flight_data/CYQR_flight_Data'

# Check if the folder path exists
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"The system cannot find the path specified: '{folder_path}'")

# Gather the first 64 .csv files from the specified folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')][:64]
file_paths = [os.path.join(folder_path, filename) for filename in file_list]

# Read and concatenate all data from selected files
dataframes = [pd.read_csv(file_path) for file_path in file_paths]
data = pd.concat(dataframes, ignore_index=True)

# Define fuel flow parameters
fuel_params = ['ff_1_kgps', 'ff_2_kgps', 'ff_3_kgps', 'ff_4_kgps']

# Compute total fuel flow across the fuel flow parameters
data['total_fuel_flow_kgps'] = data[fuel_params].sum(axis=1)

# Compute the correlation matrix and extract correlations with total fuel flow
correlation_matrix = data.corr()
fuel_correlations = correlation_matrix['total_fuel_flow_kgps'].drop(index=fuel_params + ['total_fuel_flow_kgps'])

# Sort correlations in descending order, excluding NaN values
sorted_correlations = fuel_correlations.sort_values(ascending=False).dropna()

# Save sorted correlations and relevant subset of the correlation matrix to an Excel file
excel_file = 'coef_correlation_input.xlsx'
with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
    # Write sorted correlations to the first sheet
    sorted_correlations.reset_index().rename(columns={'index': 'Parameter', 'total_fuel_flow_kgps': 'Correlation'}).to_excel(writer, sheet_name='Fuel_Correlations', index=False)
    
    # Write a subset of the correlation matrix to the second sheet
    correlation_matrix_subset = correlation_matrix.loc[sorted_correlations.index, ['total_fuel_flow_kgps']].dropna()
    correlation_matrix_subset.to_excel(writer, sheet_name='Correlation_Matrix_Subset')

print(f"Output saved to {excel_file}")
