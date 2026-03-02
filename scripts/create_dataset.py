import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from data_loader import load_data

def bandpass_filter(data, lowcut=0.17, highcut=0.4, fs=32.0, order=4):
    """Applies a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def create_windows_for_csv(subject_id, airflow, thoracic, events, fs=32, window_sec=30, overlap_sec=15):
    """
    Splits signals into 30s windows and formats them as flat dictionary rows for CSV export.
    Returns two lists of dictionaries: one for breathing, one for sleep stages.
    """
    window_len = int(window_sec * fs)          
    step = int((window_sec - overlap_sec) * fs) 
    
    breathing_rows = []
    sleep_stage_rows = []
    
    min_len = min(len(airflow), len(thoracic))
    window_id = 0
    
    # Generate dynamic column names for the signals
    airflow_cols = [f"Airflow_{i}" for i in range(window_len)]
    thoracic_cols = [f"Thoracic_{i}" for i in range(window_len)]
    
    for start_idx in range(0, min_len - window_len + 1, step):
        end_idx = start_idx + window_len
        
        window_airflow = airflow[start_idx:end_idx]
        window_thoracic = thoracic[start_idx:end_idx]
        
        start_time = start_idx / fs
        end_time = end_idx / fs
        
        # Default labels
        current_breathing = 0  # 0: Normal, 1: Hypopnea, 2: Apnea
        current_sleep_stage = "Unknown"
        
        for event in events:
            # Check for overlap
            if not (end_time <= event['start'] or start_time >= event['end']):
                # Breathing Logic
                if event['label'] == 'Hypopnea':
                    current_breathing = max(current_breathing, 1)
                elif event['label'] == 'Obstructive Apnea':
                    current_breathing = max(current_breathing, 2)
                
                # Sleep Stage Logic (takes the stage of the first overlapping event)
                if event['sleep_stage'] != "Unknown":
                    current_sleep_stage = event['sleep_stage']
        
        # Create base dictionary with metadata
        base_row = {
            'Subject': subject_id,
            'Window_ID': window_id,
            'Start_Time_Sec': round(start_time, 2),
            'End_Time_Sec': round(end_time, 2)
        }
        
        # Add the 1920 signal columns
        signal_data = dict(zip(airflow_cols, window_airflow))
        signal_data.update(dict(zip(thoracic_cols, window_thoracic)))
        
        # --- Build Breathing Row ---
        b_row = base_row.copy()
        b_row['Breathing_Label'] = current_breathing
        b_row.update(signal_data)
        breathing_rows.append(b_row)
        
        # --- Build Sleep Stage Row ---
        s_row = base_row.copy()
        s_row['Sleep_Stage'] = current_sleep_stage
        s_row.update(signal_data)
        sleep_stage_rows.append(s_row)
        
        window_id += 1
        
    return breathing_rows, sleep_stage_rows

def process_and_export_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    subject_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    all_breathing_data = []
    all_sleep_stage_data = []
    
    for subject in subject_folders:
        folder_path = os.path.join(input_dir, subject)
        print(f"Processing Subject: {subject}...")
        
        try:
            _, airflow, thoracic, _, _, events, _ = load_data(folder_path)
            
            filtered_airflow = bandpass_filter(airflow)
            filtered_thoracic = bandpass_filter(thoracic)
            
            b_rows, s_rows = create_windows_for_csv(
                subject_id=subject, 
                airflow=filtered_airflow, 
                thoracic=filtered_thoracic, 
                events=events
            )
            
            all_breathing_data.extend(b_rows)
            all_sleep_stage_data.extend(s_rows)
            print(f"  -> Extracted {len(b_rows)} windows.")
            
        except Exception as e:
            print(f"  -> Failed to process {subject}: {e}")
            
    print("\nConverting to DataFrames and saving to CSV... (This might take a minute)")
    
    # Export Breathing Dataset
    df_breathing = pd.DataFrame(all_breathing_data)
    breathing_path = os.path.join(output_dir, "breathing_dataset.csv")
    df_breathing.to_csv(breathing_path, index=False)
    print(f"Saved: {breathing_path}")
    
    # Export Sleep Stage Dataset
    df_sleep = pd.DataFrame(all_sleep_stage_data)
    sleep_path = os.path.join(output_dir, "sleep_stage_dataset.csv")
    df_sleep.to_csv(sleep_path, index=False)
    print(f"Saved: {sleep_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', type=str, default="Data", help="Path to main directory")
    parser.add_argument('-output', type=str, default="Dataset", help="Directory to save CSVs")
    args = parser.parse_args()
    
    process_and_export_csv(args.input, args.output)