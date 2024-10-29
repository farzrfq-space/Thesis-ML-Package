import pandas as pd
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

# Appendices
# Appendix C: Visualizing Altitude Profiles with Key Timepoints for Flight Data

# Load the database of timepoints
timepoints_df = pd.read_excel("database_timepoints_new.xlsx")

# Define folder path containing the flight data files
folder_path = "flight data/new-data-NASA-FUEL"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Process each file for altitude profile visualization
for file in all_files:
    filename = file.split('.')[0]
    timepoints_row = timepoints_df[timepoints_df['filename'] == filename]
    
    # Skip if no timepoints data available for this file
    if timepoints_row.empty:
        continue
    timepoints_row = timepoints_row.iloc[0]
    
    # Load flight data CSV
    file_path = os.path.join(folder_path, file)
    flight_data = pd.read_csv(file_path)

    # Plot altitude profiles with key timepoints
    plt.figure(figsize=(10, 6))
    plt.plot(flight_data['time_s'], flight_data['hbaro_m'], label='hbaro_m', color='blue')
    plt.plot(flight_data['time_s'], flight_data['hralt_m'], label='hralt_m', color='red')

    # Mark important timepoints in the plot
    for tp_label, color in zip(['tp101', 'tp102', 'tp105', 'tp106', 'tp107'], ['green']*5):
        tp = timepoints_row[tp_label]
        if not pd.isnull(tp):
            idx = flight_data.index[flight_data['time_s'] == tp][0]
            plt.scatter(tp, flight_data.loc[idx, 'hralt_m'], color=color, label=tp_label)

    # Mark tp103 and tp104 with different colors
    for tp_label, color, column in [('tp103', 'orange', 'hbaro_m'), ('tp104', 'purple', 'hbaro_m')]:
        tp = timepoints_row[tp_label]
        if not pd.isnull(tp):
            idx = flight_data.index[flight_data['time_s'] == tp][0]
            plt.scatter(tp, flight_data.loc[idx, column], color=color, label=tp_label)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Altitude (meters)')
    plt.title(f'Altitude over Time for {filename}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Appendix D: Visualization and Calculation of Fuel Consumption Across Flight Phases

# Calculate total fuel consumption for a given phase
def calculate_total_fuel(flight_data, start_tp, end_tp):
    fuel_flow_columns = ['ff_1_kgps', 'ff_2_kgps', 'ff_3_kgps', 'ff_4_kgps']
    total_fuel = flight_data.loc[start_tp:end_tp, fuel_flow_columns].sum().sum()
    return total_fuel

# Process a single file for fuel consumption per phase
def process_file(row):
    filename = row['filename']
    file_path = os.path.join(folder_path, f'{filename}.csv')
    
    if os.path.exists(file_path):
        # Load flight data
        flight_data = pd.read_csv(file_path)
        
        # Calculate fuel consumption across all phases
        total_fuel_takeoff = calculate_total_fuel(flight_data, row['tp101'], row['tp102'])
        total_fuel_climb = calculate_total_fuel(flight_data, row['tp102'], row['tp103'])
        total_fuel_cruise = calculate_total_fuel(flight_data, row['tp103'], row['tp104'])
        total_fuel_descent = calculate_total_fuel(flight_data, row['tp104'], row['tp105'])
        total_fuel_approach = calculate_total_fuel(flight_data, row['tp105'], row['tp106'])
        total_fuel_final_approach = calculate_total_fuel(flight_data, row['tp106'], row['tp107'])
        total_fuel_landing = calculate_total_fuel(flight_data, row['tp107'], len(flight_data) - 1)

        # Summarize total fuel per file
        total_fuel_per_file = (
            total_fuel_takeoff + total_fuel_climb + total_fuel_cruise +
            total_fuel_descent + total_fuel_approach + total_fuel_final_approach +
            total_fuel_landing
        )

        return {
            'filename': filename,
            'total_fuel_takeoff': total_fuel_takeoff,
            'total_fuel_climb': total_fuel_climb,
            'total_fuel_cruise': total_fuel_cruise,
            'total_fuel_descent': total_fuel_descent,
            'total_fuel_approach': total_fuel_approach,
            'total_fuel_final_approach': total_fuel_final_approach,
            'total_fuel_landing': total_fuel_landing,
            'total_fuel_per_file': total_fuel_per_file
        }
    return None

# Multiprocessing for faster processing of multiple files
if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_file, [row for index, row in timepoints_df.iterrows()])

    # Filter out empty results and save to DataFrame
    results = [result for result in results if result is not None]
    results_df = pd.DataFrame(results)

    # Write results to output file
    with open('output.txt', 'w') as outfile:
        outfile.write("Fuel Consumption Per Flight Phase:\n")
        outfile.write(results_df.to_string(index=False))

        # Calculate average fuel consumption per phase
        numeric_columns = [
            'total_fuel_takeoff', 'total_fuel_climb', 'total_fuel_cruise',
            'total_fuel_descent', 'total_fuel_approach', 'total_fuel_final_approach',
            'total_fuel_landing'
        ]
        avg_fuel_consumption = results_df[numeric_columns].mean()
        outfile.write("\n\nAverage Fuel Consumption Per Phase:\n")
        outfile.write(avg_fuel_consumption.to_string())

    # Plot average fuel consumption per flight phase
    phases = [
        'Takeoff', 'Climb', 'Cruise', 'Descent', 'Approach', 'Final Approach', 'Landing'
    ]
    avg_consumption_values = avg_fuel_consumption.values

    plt.figure(figsize=(12, 8))
    bars = plt.bar(phases, avg_consumption_values, color='#1f77b4', edgecolor='black', linewidth=1.2)

    plt.xlabel('Flight Phase', fontsize=14, labelpad=10)
    plt.ylabel('Average Fuel Consumption (kg)', fontsize=14, labelpad=10)
    plt.title('Average Fuel Consumption per Flight Phase for Flights from CYQR to KMSP', fontsize=16, pad=15)
    
    # Add labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + yval * 0.01, round(yval, 2), ha='center', va='bottom', fontsize=12)

    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('average_fuel_consumption_per_phase.png')
    plt.show()

# Output messages
print("Total fuel per flight phase saved to output.txt.")
print("Average fuel consumption per flight phase plot saved as average_fuel_consumption_per_phase.png.")
