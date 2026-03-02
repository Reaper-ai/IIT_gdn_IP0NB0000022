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
#### ⚠️ IMPORTANT PREREQUISITE: Due to GitHub's strict 100 MB file size limit, the processed datasets (breathing_dataset.csv and sleep_stage_dataset.csv) are not included in this repository. You must run the create_dataset.py script (as detailed in Section 1) to generate these files locally before attempting to train the model.

Once datasets are generated, proceed with training the 1D Convolutional Neural Network (CNN). The complete training pipeline is contained within the provided Jupyter Notebook.

What the notebook does:

- **Data Loading**: Uses a custom PyTorch Dataset to efficiently load the flattened CSV rows and reshape them into (Channels, Sequence_Length) tensors.
- **Model Initialization**: Constructs a 1D CNN optimized for dual-channel physiological time-series data using Conv1d, BatchNorm1d, and AdaptiveAvgPool1d layers.
- **LOOCV Strategy**: Executes a strict Leave-One-Participant-Out Cross-Validation loop to ensure the model's performance generalizes to unseen patients.
- **Imbalance Handling**: Dynamically calculates and applies class weights to the Cross-Entropy Loss function to prevent the model from overfitting to the majority "Normal" breathing class.
- **Metric Visualization**: Outputs per-fold confusion matrices via Seaborn and a comprehensive final bar chart comparing Accuracy, Precision, Recall, and F1-Score across all participants.

**How to run**:
Simply open the notebook in your preferred environment (Jupyter, VS Code, Google Colab) and execute the cells sequentially:

```bash
jupyter notebook model_training.ipynb
```