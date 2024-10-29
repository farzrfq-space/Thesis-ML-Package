import pandas as pd 
import numpy as np 
import os 
import time 
import joblib 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import r2_score 

# Measure the start time 
start_time = time.time() 

# Load the pre-trained model and scaler
model_path = 'mlp_model_total.joblib' 
model = joblib.load(model_path) 
scaler_path = 'scaler_total.joblib' 
scaler = joblib.load(scaler_path) 

# Set data folder and output Excel writer
data_folder = 'flight data/validation' 
excel_writer = pd.ExcelWriter('validation_metrics_per_file.xlsx', engine='xlsxwriter') 

# Get list of CSV files in the data folder
file_list = os.listdir(data_folder) 
csv_files = [f for f in file_list if f.endswith('.csv')] 

# Process each CSV file 
for filename in csv_files: 
    print(f"\nProcessing file: {filename}") 
    try: 
        # Construct the full path to the new data 
        new_data_path = os.path.join(data_folder, filename) 
        new_data = pd.read_csv(new_data_path, header=0) 

        input_columns = [ 
            'n11_rpm', 'n12_rpm', 'n13_rpm', 'n14_rpm', 
            'tas_mps', 'press_static_npm2', 'hdot_2_mps', 
            'aoac_rad', 'hbaro_m', 'temp_static_deg', 
            'flap_te_pos',  
        ] 

        # Select input features from new data 
        X_new = new_data[input_columns] 
        X_new_scaled = scaler.transform(X_new)  # Normalize new data 

        # Predict fuel flow using the loaded model 
        y_pred = model.predict(X_new_scaled) 
        actual_total_fuel_flow = new_data[['ff_1_kgps', 'ff_2_kgps', 'ff_3_kgps', 'ff_4_kgps']].sum(axis=1) 

        timepoints_df = pd.read_excel('database_timepoints_new.xlsx', sheet_name='flight_N103770') 
        filename_no_ext = os.path.splitext(filename)[0] 
        timepoints_row = timepoints_df[timepoints_df['filename'] == filename_no_ext].iloc[0] 

        # Initialize lists to store metrics
        rmse_kgps_per_phase = [] 
        mae_kgps_per_phase = [] 
        mre_per_phase = [] 
        r2_per_phase = [] 
        actual_total_per_phase = [] 
        predicted_total_per_phase = [] 
        num_samples_per_phase = [] 

        phases = ['takeoff', 'climb', 'cruise', 'descent', 'approach', 'final_approach', 'landing'] 
        start_timepoints = ['tp101', 'tp102', 'tp103', 'tp104', 'tp105', 'tp106', 'tp107'] 
        end_timepoints = ['tp102', 'tp103', 'tp104', 'tp105', 'tp106', 'tp107', 'tp108'] 

        # Calculate metrics for each flight phase 
        for phase, start_tp, end_tp in zip(phases, start_timepoints, end_timepoints): 
            start_tp = timepoints_row[start_tp] 
            end_tp = timepoints_row[end_tp] if isinstance(end_tp, str) else end_tp

            indices = range(start_tp, end_tp + 1) 
            num_samples_phase = len(indices) 

            residuals = actual_total_fuel_flow.values[indices] - y_pred[indices] 
            rmse_kgps_phase = np.sqrt(np.mean(residuals**2)) 
            mae_kgps_phase = np.mean(np.abs(residuals)) 

            # Calculate MRE only for non-zero actual values
            non_zero_actual = actual_total_fuel_flow.values[indices] != 0 
            mre_phase = np.mean(np.abs(residuals[non_zero_actual] / actual_total_fuel_flow.values[indices][non_zero_actual])) * 100 
            r2_phase = r2_score(actual_total_fuel_flow.values[indices], y_pred[indices]) 

            # Append metrics to lists 
            rmse_kgps_per_phase.append(rmse_kgps_phase) 
            mae_kgps_per_phase.append(mae_kgps_phase) 
            mre_per_phase.append(mre_phase) 
            r2_per_phase.append(r2_phase) 
            num_samples_per_phase.append(num_samples_phase) 

            actual_total_phase = np.sum(actual_total_fuel_flow.values[indices]) 
            predicted_total_phase = np.sum(y_pred[indices]) 

            actual_total_per_phase.append(actual_total_phase) 
            predicted_total_per_phase.append(predicted_total_phase) 

            # Print metrics for each phase 
            print(f"\nFlight Phase: {phase}") 
            print(f"RMSE in kgps: {rmse_kgps_phase:.4f}") 
            print(f"MAE in kgps: {mae_kgps_phase:.4f}") 
            print(f"MRE in %: {mre_phase:.2f}") 
            print(f"R²: {r2_phase:.4f}") 
            print(f"Number of Samples: {num_samples_phase}") 

        # Convert RMSE and MAE from kgps to kg/h 
        rmse_kgh_per_phase = [rmse * 3600 for rmse in rmse_kgps_per_phase] 
        mae_kgh_per_phase = [mae * 3600 for mae in mae_kgps_per_phase] 

        overall_r2 = r2_score(actual_total_fuel_flow, y_pred) 

        # Store metrics and total fuel consumption per phase for this file 
        metrics_per_file_df = pd.DataFrame({ 
            'Flight_Phase': phases, 
            'Num_Samples': num_samples_per_phase, 
            'RMSE_kgps': rmse_kgps_per_phase, 
            'MAE_kgps': mae_kgps_per_phase, 
            'MRE_%': mre_per_phase, 
            'R²': r2_per_phase,
            'RMSE_kgh': rmse_kgh_per_phase, 
            'MAE_kgh': mae_kgh_per_phase, 
            'Actual_Total_Fuel': actual_total_per_phase, 
            'Predicted_Total_Fuel': predicted_total_per_phase 
        }) 

        # Add the overall R² to the metrics dataframe 
        overall_metrics_df = pd.DataFrame({ 
            'Metric': ['Overall R²'], 
            'Value': [overall_r2] 
        }) 

        # Save metrics to Excel 
        metrics_per_file_df.to_excel(excel_writer, sheet_name=filename_no_ext, index=False) 
        overall_metrics_df.to_excel(excel_writer, sheet_name=f"{filename_no_ext}_overall", index=False) 

    except Exception as e: 
        print(f"Error processing file {filename}: {e}") 

# Save the Excel writer 
excel_writer._save() 

# Measure the end time 
end_time = time.time() 
processing_time = end_time - start_time 
print(f"\nTotal processing time: {processing_time:.2f} seconds")
