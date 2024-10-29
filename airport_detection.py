import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from multiprocessing import Pool

# Paths to the data files
timepoints_file = 'database_timepoints.xlsx'
airports_file = 'airports.csv'
flight_data_folder = 'flight_data/new-data-NASA-FUEL'
output_csv_file = 'output_airport_detection.csv'

# Load data
timepoints_df = pd.read_excel(timepoints_file)
airports_df = pd.read_csv(airports_file)

# Conversion functions
def rad_to_deg(rad):
    return degrees(rad)

def deg_to_rad(deg):
    return radians(deg)

# Haversine formula to calculate the distance between two points on Earth
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Radius of the Earth in kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

# Function to detect departure and arrival airports
def detect_airports(filename):
    file_path = os.path.join(flight_data_folder, f"{filename}.csv")
    flight_data = pd.read_csv(file_path)
    
    # Filter for filename in timepoints data
    timepoints_row = timepoints_df[timepoints_df['filename'] == filename]
    if timepoints_row.empty:
        return None
    
    timepoints_row = timepoints_row.iloc[0]
    tp101, tp107 = timepoints_row['tp101'], timepoints_row['tp107']
    
    # Extract start and end positions in radians
    start_position = flight_data.loc[tp101, ['lon_rad', 'lat_rad']]
    end_position = flight_data.loc[tp107, ['lon_rad', 'lat_rad']]
    
    # Convert positions from radians to degrees
    start_position_deg = start_position.apply(rad_to_deg)
    end_position_deg = end_position.apply(rad_to_deg)
    
    # Calculate distances to each airport from start and end positions
    airport_positions = airports_df[['longitude_deg', 'latitude_deg']]
    start_distances = airport_positions.apply(lambda row: haversine(
        start_position_deg['lon_rad'], start_position_deg['lat_rad'],
        row['longitude_deg'], row['latitude_deg']), axis=1)
    end_distances = airport_positions.apply(lambda row: haversine(
        end_position_deg['lon_rad'], end_position_deg['lat_rad'],
        row['longitude_deg'], row['latitude_deg']), axis=1)
    
    # Get nearest airports by index
    start_airport_idx = start_distances.idxmin()
    end_airport_idx = end_distances.idxmin()
    
    # Extract airport names
    departure_airport = airports_df.iloc[start_airport_idx]['name']
    arrival_airport = airports_df.iloc[end_airport_idx]['name']
    
    return filename, departure_airport, arrival_airport

if __name__ == '__main__':
    # Process each flight data file
    files_to_process = [filename.split(".")[0] for filename in 
                        os.listdir(flight_data_folder) if filename.endswith(".csv")]

    with Pool() as pool:
        results = pool.map(detect_airports, files_to_process)

    # Save results to CSV
    df = pd.DataFrame(results, columns=['filename', 'departure_airport', 'arrival_airport'])
    df.to_csv(output_csv_file, index=False)
