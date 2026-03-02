import os
import numpy as np
from datetime import datetime, timedelta

def parse_continuous_data(filepath):
    """Extracts signal values from the continuous data text files."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Actual data begins at line 8 (index 7)
    data_lines = lines[7:]
    
    values = []
    for line in data_lines:
        try:
            # Format: '30.05.2024 20:59:00,000; 120' -> extract '120'
            # Replace comma with dot just in case European decimal format is used for values
            val_str = line.strip().split(';')[1].replace(',', '.')
            values.append(float(val_str))
        except (IndexError, ValueError):
            continue
            
    return np.array(values)

def get_global_start_time(filepath):
    """Extracts the overall start time of the recording from the header."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("Start Time:"):
                # Handle potential trailing characters and extract the date string
                time_str = line.split("Start Time: ")[1].strip()
                try:
                    # Attempt to parse format like: 5/30/2024 8:59:00 PM
                    return datetime.strptime(time_str, "%m/%d/%Y %I:%M:%S %p")
                except ValueError:
                    # Fallback for standard ISO or other formats if they vary
                    pass
    # If not found in header, fallback to parsing the first data line
    with open(filepath, 'r') as f:
        lines = f.readlines()
        first_data_line = lines[7]
        time_str = first_data_line.split(';')[0].strip()
        return datetime.strptime(time_str, "%d.%m.%Y %H:%M:%S,%f")

def parse_events(filepath, global_start_time):
    """Parses the Flow Events file, extracting both Breathing Labels and Sleep Stages."""
    events = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Data begins at line 6 (index 5)
    data_lines = lines[5:]
    
    for line in data_lines:
        if not line.strip(): continue
        
        parts = line.strip().split(';')
        if len(parts) < 3: continue
            
        time_range = parts[0].strip()
        label_str = parts[2].strip()
        
        # Extract sleep stage if the 4th column exists, otherwise label it 'Unknown'
        sleep_stage = parts[3].strip() if len(parts) > 3 else "Unknown"
        
        try:
            start_str, end_time_str = time_range.split('-')
            
            dt_start = datetime.strptime(start_str, "%d.%m.%Y %H:%M:%S,%f")
            end_time_obj = datetime.strptime(end_time_str, "%H:%M:%S,%f").time()
            dt_end = datetime.combine(dt_start.date(), end_time_obj)
            
            if dt_end < dt_start:
                dt_end += timedelta(days=1)
                
            start_sec = (dt_start - global_start_time).total_seconds()
            end_sec = (dt_end - global_start_time).total_seconds()
            
            # Keep ALL events now so we can track Sleep Stages across the whole night
            events.append({
                'start': start_sec,
                'end': end_sec,
                'label': label_str,
                'sleep_stage': sleep_stage
            })
        except Exception as e:
            print(f"Skipping malformed event line: {line.strip()} -> {e}")
            
    return events

def load_data(subject_folder):
    """
    Main function to load and synchronize all data for a single subject.
    Expects folder path like 'Data/AP01'
    """
    flow_file = thorac_file = spo2_file = events_file = None
    
    # Auto-detect files using keyword matching to handle naming inconsistencies 
    for f in os.listdir(subject_folder):
        # Match "Flow Events" first so it doesn't get captured by the general "Flow" check
        if 'Flow Events' in f: 
            events_file = os.path.join(subject_folder, f)
        # Now it's safe to check for general Flow data (matches "Flow -", "Flow Signal -", "Flow Nasal -")
        elif 'Flow' in f:
            flow_file = os.path.join(subject_folder, f)
        # Matches "Thorac -", "Thorac Signal -", "Thorac Movement -"
        elif 'Thorac' in f:
            thorac_file = os.path.join(subject_folder, f)
        # Matches "SPO2 -", "SPO2 Signal -"
        elif 'SPO2' in f:
            spo2_file = os.path.join(subject_folder, f)
        
    if not all([flow_file, thorac_file, spo2_file, events_file]):
        raise FileNotFoundError(
            f"Missing one or more required files in {subject_folder}.\n"
            f"Found -> Flow: {flow_file}, Thorac: {thorac_file}, SpO2: {spo2_file}, Events: {events_file}"
        )
        
    global_start = get_global_start_time(flow_file)
    
    # Parse signals into numpy arrays
    airflow = parse_continuous_data(flow_file)
    thoracic = parse_continuous_data(thorac_file)
    spo2 = parse_continuous_data(spo2_file)
    
    # Generate time vectors efficiently using the known sample rates (32 Hz and 4 Hz)
    time_32hz = np.arange(len(airflow)) / 32.0
    time_4hz = np.arange(len(spo2)) / 4.0
    
    # Parse annotated events
    events = parse_events(events_file, global_start)
    
    # ADD global_start TO THIS RETURN STATEMENT:
    return time_32hz, airflow, thoracic, time_4hz, spo2, events, global_start