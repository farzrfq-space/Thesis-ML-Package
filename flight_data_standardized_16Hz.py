import os
import pandas as pd
import multiprocessing as mp

folder_path = "flight data"
sampling_rate_reduction_factor = 16

def standardize_csv(filename):
    file_path = os.path.join(folder_path, filename)
    data = pd.read_csv(file_path)

    standardized_data = pd.DataFrame()
    standardized_data = pd.concat([standardized_data, data.iloc[[0]]], ignore_index=True)  # Keep first row

    for index, row in data.iloc[1:].iterrows():
        if index % sampling_rate_reduction_factor == 0 and row['time_s'] % 1 == 0:
            standardized_data = pd.concat([standardized_data, row.to_frame().transpose()], ignore_index=True)

    standardized_data.to_csv(file_path, index=False)  # Overwrite the file
    print(f"Finished processing {filename}")

if __name__ == '__main__':
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".csv")]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(standardize_csv, csv_files)

    print("All CSV files have been processed.")
