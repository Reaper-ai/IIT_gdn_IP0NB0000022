# IIT-GDN Sleep Study Analysis

This project is designed to process, analyze, and visualize sleep study data. It includes scripts for loading raw signal data, generating processed datasets for machine learning, and creating detailed signal visualizations.

## Project Structure

```
.
├── Data/                   # Raw data folders (e.g., AP01, AP02) containing signal text files
├── Dataset/                # Output folder for processed CSV datasets
├── Visualizations/         # Output folder for generated PDF visualizations
├── models/
│   └── model.ipynb         # Jupyter notebook for model development
├── scripts/
│   ├── create_dataset.py   # Script to process raw data into CSV datasets
│   ├── data_loader.py      # Utility functions for parsing raw data files
│   └── vis.py              # Script to visualize signals
├── pyproject.toml          # Project dependencies and configuration
└── README.md
```

## Setup and Installation

1. **Prerequisites**: Ensure you have Python 3.11 or later installed.

2. **Install Dependencies**:
   This project uses `pyproject.toml` for dependency management. You can install the required packages using pip:

   ```bash
   pip install .
   ```
   
   Or manually install the dependencies listed in `pyproject.toml`:
   ```bash
   pip install matplotlib numpy pandas scikit-learn scipy seaborn torch
   ```

## Usage

### 1. Generating Datasets
The `create_dataset.py` script processes raw text files from the `Data/` directory, applies bandpass filtering to signals, segments them into 30-second windows, and exports the data to CSV files.

```bash
# Default usage (reads from 'Data/' and saves to 'Dataset')
python scripts/create_dataset.py

# Custom input and output directories
python scripts/create_dataset.py -input Data -output Dataset
```

This will generate:
- `breathing_dataset.csv`: Contains airflow and thoracic signals labeled with breathing events (Normal, Hypopnea, Apnea).
- `sleep_stage_dataset.csv`: Contains signals labeled with sleep stages.

### 2. Visualizing Signals
The `vis.py` script generates a multi-page PDF visualization for a specific subject's sleep study data, broken down into 10-minute intervals.

```bash
# Visualize a specific subject (e.g., AP01)
python scripts/vis.py -name Data/AP01
```

The output PDF will be saved in the `Visualizations/` directory (e.g., `Visualizations/AP01_visualization.pdf`).

### 3. Data Loading API
The `scripts/data_loader.py` module provides the `load_data` function, which can be imported and used in other scripts or notebooks to retrieve raw signals and events.

### 4. Model training