import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta
from data_loader import load_data

def visualize_signals(subject_folder):
    # Extract the subject ID from the folder path (e.g., 'Data/AP20' -> 'AP20')
    subject_id = os.path.basename(os.path.normpath(subject_folder))
    
    print(f"Loading data from '{subject_folder}' for Subject {subject_id}...")
    try:
        # Now we unpack global_start directly from our robust dataloader
        time_32hz, airflow, thoracic, time_4hz, spo2, events, global_start = load_data(subject_folder)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Generating multi-page plot (10-minute intervals)...")
    
    # Ensure the output directory exists
    output_dir = 'Visualizations'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'{subject_id}_visualization.pdf')
    
    # Convert time from seconds to minutes for the X-axis
    time_32hz_mins = time_32hz / 60.0
    time_4hz_mins = time_4hz / 60.0
    
    # Determine the total duration in minutes
    max_time_mins = max(time_32hz_mins[-1] if len(time_32hz_mins) > 0 else 0, 
                        time_4hz_mins[-1] if len(time_4hz_mins) > 0 else 0)
    
    # Calculate total pages needed for 10-minute intervals
    interval_mins = 10
    total_pages = math.ceil(max_time_mins / interval_mins)
    
    event_colors = {'Hypopnea': 'orange', 'Obstructive Apnea': 'purple'}

    # Use PdfPages to create a multi-page PDF
    with PdfPages(output_filename) as pdf:
        for page in range(total_pages):
            start_min = page * interval_mins
            end_min = start_min + interval_mins
            
            print(f"  -> Rendering Page {page + 1} of {total_pages}...")
            
            mask_32 = (time_32hz_mins >= start_min) & (time_32hz_mins < end_min)
            mask_4 = (time_4hz_mins >= start_min) & (time_4hz_mins < end_min)
            
            if not np.any(mask_32) and not np.any(mask_4):
                continue
                
            fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
            
            # 1. Nasal Airflow
            axs[0].plot(time_32hz_mins[mask_32], airflow[mask_32], color='#1f77b4', linewidth=0.8, label='Nasal Airflow')
            axs[0].set_ylabel('Amplitude')
            axs[0].legend(loc='upper right')
            axs[0].grid(True, alpha=0.3)
            
            # 2. Thoracic Movement
            axs[1].plot(time_32hz_mins[mask_32], thoracic[mask_32], color='#2ca02c', linewidth=0.8, label='Thoracic Movement')
            axs[1].set_ylabel('Amplitude')
            axs[1].legend(loc='upper right')
            axs[1].grid(True, alpha=0.3)
            
            # 3. SpO2
            axs[2].plot(time_4hz_mins[mask_4], spo2[mask_4], color='#d62728', linewidth=1.5, label='SpO2 (%)')
            axs[2].set_ylabel('SpO2 Level')
            axs[2].set_xlabel(f'Time (Minutes from Start of Recording)')
            axs[2].legend(loc='upper right')
            axs[2].grid(True, alpha=0.3)
            
            # Overlay annotations that overlap with the current 10-minute window
            target_labels = ['Hypopnea', 'Obstructive Apnea'] # Only plot abnormalities
            
            for event in events:
                label = event['label']
                
                # SKIP any event that is not in our target list
                if label not in target_labels:
                    continue
                    
                e_start_min = event['start'] / 60.0
                e_end_min = event['end'] / 60.0
                
                if e_end_min > start_min and e_start_min < end_min:
                    color = event_colors[label]
                    for ax in axs:
                        ax.axvspan(e_start_min, e_end_min, color=color, alpha=0.4, label=label)
            
            handles, labels = axs[0].get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            axs[0].legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
            
            axs[2].set_xlim(start_min, end_min)
            
            # Calculate absolute datetimes for the title safely
            if global_start:
                page_start_dt = global_start + timedelta(minutes=start_min)
                page_end_dt = global_start + timedelta(minutes=end_min)
                time_format = "%I:%M:%S %p"
                title_str = f'Sleep Study Signals - Subject {subject_id}\nTimestamp: {page_start_dt.strftime(time_format)} to {page_end_dt.strftime(time_format)}'
            else:
                title_str = f'Sleep Study Signals - Subject {subject_id}\nMinutes: {start_min} to {end_min}'
                
            plt.suptitle(title_str, fontsize=16)
            plt.tight_layout()
            
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
    print(f"Saved 10-minute interval visualization to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Sleep Study Signals")
    parser.add_argument('-name', type=str, required=True, help="Path to the subject data folder (e.g., Data/AP20)")
    args = parser.parse_args()
    
    visualize_signals(args.name)