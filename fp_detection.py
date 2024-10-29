import pandas as pd
import os
from multiprocessing import Pool

folder_path = "flight data/flight_data"

def process_file(filename):
    if filename.endswith(".csv"):
        # Read the data
        data = pd.read_csv(os.path.join(folder_path, filename)).dropna()

        # Downsampling the data to 1 Hz
        data = data.iloc[::20, :].reset_index(drop=True)  # Assuming original data is at 20 Hz

        # Conditions for Takeoff
        condition_takeoff = (
            (data['hbaro_m'] > 30) &  # Above 30 meters
            (data['hdot_1_mps'] > 2) &  # Vertical speed greater than 2 m/s
            (data['cas_mps'] > 70)  # Airspeed greater than 70 m/s
        )
        takeoff_index = data.loc[condition_takeoff].index[0] if not data.loc[condition_takeoff].empty else None

        # Conditions for Climb
        condition_climb = (
            (data['hbaro_m'] > 300) &  # Above 300 meters
            (data['hdot_1_mps'] > 2)  # Sustained climb rate
        )
        climb_index = data.loc[condition_climb].index[0] if not data.loc[condition_climb].empty else None

        # Conditions for Cruise
        condition_cruise = (
            (data['hbaro_m'] > 1000) &  # Above 1000 meters
            (data['hdot_1_mps'] < 1) &  # Vertical speed less than 1 m/s
            (data['cas_mps'] > 200)  # Airspeed greater than 200 m/s
        )
        cruise_index = data.loc[condition_cruise].index[0] if not data.loc[condition_cruise].empty else None

        # Conditions for Descent
        condition_descent = (
            (data['hbaro_m'] < 1000) &  # Below 1000 meters
            (data['hdot_1_mps'] < -1) &  # Negative vertical speed (descending)
            (data['cas_mps'] < 200)  # Airspeed less than 200 m/s
        )
        descent_index = data.loc[condition_descent].index[0] if not data.loc[condition_descent].empty else None

        # Conditions for Approach
        condition_approach = (
            (data['hbaro_m'] < 300) &  # Below 300 meters
            (data['hdot_1_mps'] < -0.5) &  # Negative vertical speed (descending)
            (data['cas_mps'] < 70)  # Airspeed less than 70 m/s
        )
        approach_index = data.loc[condition_approach].index[0] if not data.loc[condition_approach].empty else None

        # Final Approach
        if approach_index is not None:
            data_trimmed = data[data.index >= approach_index]

            condition_final_approach = (
                (data_trimmed['hbaro_m'] < 300) &
                (data_trimmed['hdot_1_mps'] < -0.5) &
                (data_trimmed['cas_mps'] < 70)
            )
            final_approach_index = data_trimmed.loc[condition_final_approach].index[0] if not data_trimmed.loc[condition_final_approach].empty else None
        else:
            final_approach_index = None

        # Landing
        if final_approach_index is not None:
            data_trimmed = data[data.index >= final_approach_index]

            condition_landing = (
                (data_trimmed['hbaro_m'] <= 0) &  # At or below ground level
                (data_trimmed['hdot_1_mps'] < -2)  # A steady descent
            )
            landing_index = data_trimmed.loc[condition_landing].index[0] if not data_trimmed.loc[condition_landing].empty else None
        else:
            landing_index = None

        return (filename, takeoff_index, climb_index, cruise_index, descent_index, approach_index, final_approach_index, landing_index)

if __name__ == '__main__':
    files = os.listdir(folder_path)
    with Pool() as pool:
        results = pool.map(process_file, files)

    results_df = pd.DataFrame(results, columns=['filename', 'takeoff_index', 'climb_index', 'cruise_index', 'descent_index', 'approach_index', 'final_approach_index', 'landing_index'])
    results_df.to_csv("flight_phase_detection_results.csv", index=False)
