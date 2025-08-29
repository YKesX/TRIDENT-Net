# TRIDENT-Net

**Author**: Yağızhan Keskin

A modular multimodal fusion system for processing visible/EO, radar, and thermal/IR sensor data with explainable AI capabilities.

## Overview

TRIDENT-Net is a PyTorch-based system designed for robust multimodal sensor fusion with built-in explainability and guard mechanisms. The system processes data from three main modalities:

- **TRIDENT-I**: Visible/EO image processing (fragmentation detection, thermal attention, change detection)
- **TRIDENT-R**: Radar signal processing (micro-Doppler analysis, pulse characterization, transformer-based pattern recognition)
- **TRIDENT-T**: Thermal/IR processing (plume detection and tracking, cooling curve analysis)
- **TRIDENT-F**: Multimodal fusion (late SVM, cross-attention, fuzzy rules)
- **SpoofShield**: Guard module for consistency checks and spoofing detection

## Key Features

- **Modular Architecture**: Individual components can be trained and deployed independently
- **Multiple Fusion Strategies**: SVM, transformer-based cross-attention, and rule-based fusion
- **Explainable AI**: Attention maps, event tokens, and textual explanations
- **Guard Mechanisms**: Physics-based consistency checks and plausibility assessment
- **Real-time Inference**: REST API server for streaming inference
- **Comprehensive Evaluation**: AUROC, F1, Brier score, ECE, time-to-confidence metrics

## System Outputs

The system provides standardized outputs:

- `p_outcome ∈ [0,1]`: Probability of outcome
- `binary_outcome ∈ {0,1}`: Binary prediction  
- `explanation`: Dict with attention maps, event descriptions, and reasoning

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Additional dependencies in `requirements.txt`

### Setup

```bash
# Clone repository
git clone https://github.com/YKesX/TRIDENT-Net.git
cd TRIDENT-Net

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### 1. List Available Components and Tasks

```bash
# List all components
trident list-components --config tasks.yml

# List all tasks  
trident list-tasks --config tasks.yml
```

### 2. Generate Synthetic Data

```bash
# Create synthetic dataset for testing
trident create-synthetic-data ./data --samples 1000
```

### 3. Train Individual Components

```bash
# Train visible/EO fragment detection
trident run-task pretrain_i1 --config tasks.yml --synthetic

# Train radar echo analysis
trident run-task pretrain_r --config tasks.yml --synthetic

# Train thermal analysis
trident run-task pretrain_t --config tasks.yml --synthetic
```

### 4. Train Fusion Module

```bash
# Train cross-attention fusion (requires pretrained components)
trident run-task train_f2 --config tasks.yml --synthetic
```

### 5. Evaluate System

```bash
# Evaluate complete system
trident eval --config tasks.yml --synthetic --output results.json
```

### 6. Run Inference Server

```bash
# Start inference server
trident serve --config tasks.yml --host 0.0.0.0 --port 8000
```

## System Architecture

### Component Structure

```
trident/
├── common/           # Core types, losses, metrics, utilities
├── data/            # Dataset handling and synthetic data generation  
├── i_models/        # TRIDENT-I (Visible/EO) modules
├── r_models/        # TRIDENT-R (Radar) modules
├── t_models/        # TRIDENT-T (Thermal/IR) modules
├── fusion_guard/    # Fusion and guard modules
├── runtime/         # Training, evaluation, serving infrastructure
└── cli.py          # Command-line interface
```

### Key Components

#### TRIDENT-I (Visible/EO)
- **i1_frag_cnn.py**: U-Net-based segmentation for debris detection
- **i2_therm_att_v.py**: Temporal attention for flash/hotspot detection  
- **i3_dual_vision.py**: Siamese network for change detection

#### TRIDENT-R (Radar)
- **r1_echo_net.py**: 1D CNN for micro-Doppler analysis
- **r2_pulse_lstm.py**: BiLSTM for pulse feature processing
- **r3_radar_former.py**: Transformer for radar token sequences

#### TRIDENT-T (Thermal/IR)
- **t1_plume_net.py**: Detection and tracking of thermal signatures
- **t2_cooling_curve.py**: GRU-based cooling curve analysis

#### Fusion & Guard
- **f1_late_svm.py**: scikit-learn SVM for late fusion
- **f2_cross_attention.py**: Transformer-based multimodal fusion
- **f3_fuzzy_rules.py**: Rule-based probability adjustment
- **s_spoof_shield.py**: Consistency and plausibility checks

## Configuration

The system uses `tasks.yml` for configuration. Key sections:

### Components

```yaml
components:
  i1:
    class: trident_i.FragCNN
    kind: segmentation
    inputs: [rgb_roi: BxCxHxW]
    outputs: [mask: Bx1xHxW, zi: Bx256, events: list]
    config: {backbone: efficientnet_b0, out_dim: 256}
```

### Tasks

```yaml
tasks:
  pretrain_i1:
    run: train
    component: i1
    dataset: train
    epochs: 50
    optimizer: {name: adamw, lr: 3e-4}
    save_to: ./checkpoints/i1.pt
```

### Pipelines

```yaml
pipelines:
  joint_train:
    steps: [pretrain_i1, pretrain_i2, pretrain_i3, pretrain_r, pretrain_t, train_f2, train_f1, train_f3, train_s, eval_joint]
```

## API Usage

### Training API

```python
from trident.runtime.config import load_config, ConfigLoader
from trident.runtime.trainer import Trainer
from trident.data.dataset import create_data_loaders

# Load configuration
config = load_config("tasks.yml")
config_loader = ConfigLoader()
config_loader.config = config

# Create trainer
trainer = Trainer(config_loader)

# Create data loaders
data_loaders = create_data_loaders(config.paths.data_root)

# Train component
results = trainer.train_single_component("pretrain_i1", data_loaders)
```

### Inference API

```python
from trident.runtime.graph import create_inference_graph
from trident.data.synthetic import generate_synthetic_batch

# Create inference graph
graph = create_inference_graph(
    config=config,
    components=["i1", "i2", "r1", "f2"],
    checkpoint_map={"i1": "./checkpoints/i1.pt", ...}
)

# Generate sample data
batch = generate_synthetic_batch(batch_size=1)

# Run inference
outputs = graph.execute(batch)
outcome = outputs["f2_outcome"]

print(f"Probability: {outcome.p_outcome.item():.3f}")
print(f"Prediction: {outcome.binary_outcome.item()}")
print(f"Explanation: {outcome.explanation}")
```

### Server API

```python
import asyncio
from trident.runtime.server import TridentClient

async def run_inference():
    client = TridentClient("http://localhost:8000")
    
    # Check server health
    health = await client.health_check()
    print(f"Server status: {health.status}")
    
    # Generate sample data
    sample = await client.generate_sample()
    
    # Run inference
    result = await client.infer(sample["data"], sample["metadata"])
    print(f"Result: {result.p_outcome:.3f}")

asyncio.run(run_inference())
```

## Performance Considerations

### Model Sizes
- Individual branch modules: ~1-5M parameters each
- Fusion modules: ~2-10M parameters  
- Total system: ~20-50M parameters

### Runtime Performance
- CPU inference: ~50-200ms per sample
- GPU inference: ~10-50ms per sample
- Memory usage: ~1-4GB GPU memory

### Optimization Options
- Mixed precision training (AMP)
- Model quantization for edge deployment
- torch.compile for PyTorch 2.x acceleration

## Testing

```bash
# Run unit tests
pytest tests/

# Test with synthetic data
trident run-task pretrain_i1 --synthetic --epochs 2

# Test inference server
trident serve --config tasks.yml &
curl http://localhost:8000/health
```

## Development

### Adding New Components

1. Create component class inheriting from `BranchModule`, `FusionModule`, or `GuardModule`
2. Implement required `forward()` method
3. Add component to configuration
4. Create factory function for instantiation

### Custom Loss Functions

```python
from trident.common.losses import get_loss_fn

# Register custom loss
def my_custom_loss(pred, target):
    return F.mse_loss(pred, target)

# Use in configuration
loss_fn = get_loss_fn("custom", loss_func=my_custom_loss)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Configuration errors**: Check component class paths and input/output specifications
3. **Missing checkpoints**: Ensure pretrained components exist before fusion training
4. **Import errors**: Verify all dependencies are installed

### Debugging

```bash
# Enable verbose logging
trident run-task pretrain_i1 --verbose

# Use synthetic data for testing
trident run-task pretrain_i1 --synthetic --epochs 1
```

## Citation

```bibtex
@software{trident_net_2024,
  title={TRIDENT-Net: Modular Multimodal Fusion System},
  author={Yağızhan Keskin},
  year={2024},
  url={https://github.com/YKesX/TRIDENT-Net}
}
```

## License

MIT License - see LICENSE file for details.