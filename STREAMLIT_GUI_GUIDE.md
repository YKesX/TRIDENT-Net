# TRIDENT-Net Streamlit GUI Guide

## Overview

The TRIDENT-Net Streamlit GUI is a modern, dark-themed web interface that allows you to:
- Select training modes (train, eval, finaltrain)
- Configure dataset directories and options
- Preview dataset files (prompts.jsonl and video files)
- Monitor training/evaluation metrics in real time
- Start and stop training/evaluation processes

The GUI is located at `/trident/gui/app.py` and provides a user-friendly interface for the TRIDENT-Net multimodal fusion system.

## Prerequisites

### 1. Python Environment
- Python 3.10 or higher
- Recommended: Use a virtual environment

### 2. Install Dependencies
Install all required packages from the requirements.txt file:

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 3. Project Setup
Ensure you have the following project structure:
```
TRIDENT-Net/
├── trident/
│   ├── gui/
│   │   └── app.py          # Main Streamlit application
│   ├── cli.py             # Command-line interface
│   └── ...                # Other modules
├── tasks.yml              # Configuration file
├── requirements.txt       # Dependencies
└── README.md
```

## Running the Streamlit GUI

### Method 1: Direct Streamlit Command (Recommended)

```bash
# From the TRIDENT-Net root directory
streamlit run trident/gui/app.py
```

### Method 2: Using Python Module

```bash
# From the TRIDENT-Net root directory
python -m streamlit run trident/gui/app.py
```

### Method 3: Custom Port and Host

```bash
# Run on specific port and host
streamlit run trident/gui/app.py --server.port 8501 --server.address 0.0.0.0
```

## Using the GUI

### 1. Initial Configuration
When you open the GUI, you'll see:

- **Mode Selection**: Choose between:
  - `train`: Standard training mode
  - `eval`: Evaluation mode
  - `finaltrain`: Final training pipeline

- **Directory Settings**:
  - **Train directory**: Path to training data (default: `./Train`)
  - **Eval directory**: Path to evaluation data (default: `./Eval`)

- **Advanced Options**:
  - **Config path**: Path to tasks.yml configuration file
  - **Pipeline**: Choose pipeline variant (normal, finaltrain)
  - **Use synthetic**: Toggle to use synthetic data instead of real datasets

### 2. Dataset Preview
The GUI automatically previews your datasets:

- **Prompts Preview**: Shows content from `prompts.jsonl` files with metadata like scenarios, bearings, elevation, etc.
- **Video Files**: Lists available `.mp4` files and indicates IR video availability

### 3. Training/Evaluation Controls

- **Start Button**: Begins the selected training/evaluation process
- **Stop Button**: Terminates the running process

### 4. Real-time Monitoring
During execution, you can monitor:

- **Live Logs**: Real-time stdout from the training process
- **Metrics Charts**: Automatically parsed metrics (Loss, AUROC, F1) displayed as line charts
- **Status Updates**: Process completion status

### 5. Visualization Tabs
The right panel provides:
- **Loss & AUROC**: Live metrics visualization
- **Confusion**: Placeholder for confusion matrix (simulated)
- **Calibration**: Placeholder for reliability diagrams (simulated)

## Data Structure Requirements

### Dataset Directory Structure
Your Train and Eval directories should follow this structure:

```
Train/
├── prompts.jsonl          # JSONL file with metadata
├── video1.mp4            # RGB videos
├── video2.mp4
└── ir/                   # Optional IR videos
    ├── video1_ir.mp4
    └── video2_ir.mp4

Eval/
├── prompts.jsonl
├── video1.mp4
└── ir/
    └── video1_ir.mp4
```

### prompts.jsonl Format
Each line should be a JSON object with fields like:
```json
{
  "timestamp_utc": "2025-01-01T12:00:00Z",
  "scenario": "example_scenario",
  "selections": {
    "bearing_deg": 45.0,
    "elevation_deg": 30.0,
    "range_km": 5.0
  },
  "video": {
    "path": "video1.mp4",
    "ir_path": "ir/video1_ir.mp4",
    "fps": 24
  }
}
```

## Configuration

### tasks.yml Configuration
Ensure your `tasks.yml` file contains proper component and task definitions. The GUI will use this file to:
- Load component configurations
- Define training pipelines
- Set hyperparameters

### Environment Variables
You can set these environment variables for additional configuration:
```bash
export PYTHONUNBUFFERED=1     # For real-time log output
export PYTHONIOENCODING=utf-8 # For proper text encoding
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   # Ensure you're in the correct directory and have installed dependencies
   pip install -r requirements.txt
   ```

2. **Port already in use**:
   ```bash
   # Use a different port
   streamlit run trident/gui/app.py --server.port 8502
   ```

3. **Dataset not found**:
   - Ensure your Train/Eval directories exist
   - Check that prompts.jsonl files are present
   - Verify video file paths in prompts.jsonl

4. **Configuration errors**:
   - Verify tasks.yml file exists and is properly formatted
   - Check component class paths in configuration

5. **Process won't start**:
   - Ensure you have proper permissions to execute Python modules
   - Check that the CLI commands work independently:
     ```bash
     python -m trident.cli --help
     ```

### Performance Tips

1. **For large datasets**: Use synthetic data toggle for testing
2. **For slow loading**: Reduce batch size in tasks.yml
3. **For memory issues**: Close other applications and consider using CPU mode

## Features Overview

### Dark Theme
The GUI uses a modern Apple-inspired dark theme with:
- Dark background gradients
- Rounded panels and cards
- Accent colors for important elements
- Responsive layout

### Real-time Processing
- Live stdout parsing and display
- Automatic metric extraction from logs
- Interactive charts that update during training
- Process management with start/stop controls

### Dataset Integration
- Automatic detection of prompts.jsonl files
- Video file enumeration with IR availability checking
- Preview tables for quick data inspection

## Advanced Usage

### Custom Commands
The GUI constructs CLI commands based on your selections. For example:

**Training command**:
```bash
python -m trident.runtime.cli train --config tasks.yml --pipeline normal --jsonl ./Train/prompts.jsonl --video-root ./Train
```

**Evaluation command**:
```bash
python -m trident.runtime.cli eval --config tasks.yml --jsonl ./Eval/prompts.jsonl --video-root ./Eval
```

### Synthetic Data Mode
When "Use synthetic" is enabled, the system will:
- Generate synthetic datasets instead of loading real data
- Use faster processing for testing and development
- Skip video file requirements

## Next Steps

After setting up the GUI:

1. **Test with synthetic data first**: Enable the synthetic toggle to verify everything works
2. **Prepare your datasets**: Organize your video files and create prompts.jsonl
3. **Configure tasks.yml**: Adjust hyperparameters and component settings
4. **Run training pipelines**: Use the GUI to monitor your training processes
5. **Evaluate results**: Switch to eval mode to test your trained models

For more detailed information about the TRIDENT-Net system architecture and components, refer to the main README.md file.