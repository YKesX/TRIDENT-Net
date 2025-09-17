# TRIDENT-Net Gradio GUI Usage

This repository now includes two GUI options for TRIDENT-Net training and evaluation:

## Option 1: Streamlit GUI (Original)
```bash
# Run from the trident/gui directory
cd trident/gui
python -m streamlit run app.py
```

## Option 2: Gradio GUI (New Alternative)
```bash
# Run from the root directory
python trident_webui.py
```

## Features Available in Both GUIs

Both interfaces provide identical functionality:

### Core Configuration
- **Mode Selection**: train, eval, finaltrain
- **Directory Settings**: Train and Eval directory paths
- **Theme Toggle**: Dark/Light mode support
- **Config Path**: Path to tasks.yml configuration file
- **Pipeline Selection**: normal, finaltrain
- **Synthetic Data**: Option to use synthetic data instead of real datasets

### Data Loader Settings
- **Batch Size**: Number of samples per batch
- **Workers**: Number of data loading workers
- **Pin Memory**: Enable/disable memory pinning for faster GPU transfer

### Checkpointing
- **Policy**: both, last_epoch, best, steps, off
- **Steps Interval**: For step-based checkpointing

### Google Drive Integration
- **Local Sync**: Mirror to local Google Drive folder
- **API Upload**: Direct upload using service account credentials

### Dataset Preview
- **Train Dataset**: View prompts.jsonl and video file listings
- **Eval Dataset**: View prompts.jsonl and video file listings
- **Real-time Preview**: Click "Preview Datasets" to refresh data view

### Process Control
- **Device Selection**: CUDA (GPU) or CPU
- **Start/Stop**: Control training/evaluation processes
- **Live Logs**: Real-time process output streaming
- **Metrics Charts**: Live visualization of Loss, AUROC, F1 scores

## Key Differences

### Streamlit GUI
- Native Streamlit components and styling
- Apple-inspired dark/light themes with custom CSS
- Auto-refreshing logs and charts
- Built-in Streamlit session state management

### Gradio GUI
- Gradio components with customizable themes
- Manual refresh for logs and charts via "Refresh" button
- Matplotlib-based chart visualization
- Global state management with AppState class

## Running Both Simultaneously

Both GUIs can run simultaneously on different ports:
- Streamlit: Usually runs on port 8501
- Gradio: Configured to run on port 7860

They operate independently and don't interfere with each other.

## Dependencies

Both GUIs require their respective frameworks:
- Streamlit GUI: `pip install streamlit`
- Gradio GUI: `pip install gradio` (already included)

All other dependencies (pandas, matplotlib, etc.) are shared between both interfaces.