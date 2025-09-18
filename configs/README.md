# TRIDENT-Net Recommended Settings for Different Hardware Configurations

This directory contains optimized configurations for different hardware setups to support both training and evaluation scenarios.

## Hardware Configurations

### Training System: A100 39GB + 70GB RAM
**File:** `a100_39gb_70gb_ram_training.json`

**Use Case:** Primary training system with powerful GPU and ample memory
- **GPU:** NVIDIA A100 40GB (39GB usable VRAM)
- **CPU Memory:** 70GB available for offloading  
- **Purpose:** Full model training with all optimizations

#### Configuration Details
- **Micro-batch size:** 4 (per GPU)
- **Gradient accumulation:** 4 steps  
- **Effective batch size:** 16 (4 × 4)
- **ZeRO Stage:** 2 with CPU offload for optimizer and parameters
- **Mixed Precision:** FP16 enabled
- **Bucket sizes:** Larger (500MB) for better throughput

### Evaluation System: CPU-only + 30GB RAM  
**File:** `cpu_only_30gb_ram.json`

**Use Case:** Windows evaluation system without GPU
- **GPU:** None (CPU-only)
- **CPU Memory:** 30GB available
- **Purpose:** Model evaluation and inference

#### Configuration Details  
- **Micro-batch size:** 1 (conserve memory)
- **Gradient accumulation:** 4 steps
- **Effective batch size:** 4 (1 × 4)
- **ZeRO Stage:** 0 (disabled - no GPU)
- **Mixed Precision:** FP32 (FP16 not supported on CPU)
- **Optimizations:** Minimal checkpointing, standard AdamW optimizer

### Legacy Config: A100 39GB + 30GB RAM
**File:** `a100_39gb_30gb_cpu.json`

**Use Case:** A100 system with limited CPU memory
- **GPU:** NVIDIA A100 40GB (39GB usable VRAM) 
- **CPU Memory:** 30GB available for offloading
- **Purpose:** Training with memory constraints

## Usage Examples

### Training System (A100 + 70GB RAM)

**For Training:**
```bash
# Using DeepSpeed launcher (recommended)
deepspeed --num_gpus 1 -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --deepspeed-config configs/a100_39gb_70gb_ram_training.json \
  --use-fp16 \
  --grad-accum-steps 4 \
  --synthetic

# Using memory-efficient CLI directly  
python -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --use-fp16 \
  --grad-accum-steps 4 \
  --max-gpu-mem 39GiB \
  --cpu-mem 70GiB \
  --zero-stage 2 \
  --optimizer adamw8bit
```

### Evaluation System (CPU-only + 30GB RAM)

**For Evaluation/Inference:**
```bash
# CPU-only evaluation (Windows compatible)
python -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --deepspeed-config configs/cpu_only_30gb_ram.json \
  --use-fp16 false \
  --grad-accum-steps 4 \
  --optimizer adamw \
  --zero-stage 0 \
  --device-map balanced

# Alternative: Using GUI with CPU device selection
# Select "CPU" in the device dropdown in either GUI
```

### Legacy System (A100 + 30GB RAM)

**For Memory-Constrained Training:**
```bash
deepspeed --num_gpus 1 -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --deepspeed-config configs/a100_39gb_30gb_cpu.json \
  --use-fp16 \
  --synthetic
```

## Expected Performance

### Training System (A100 + 70GB RAM)
- **GPU VRAM:** ~36-39GB (under 39GB limit)
- **CPU Memory:** ~50-65GB (under 70GB limit)  
- **Training Speed:** Optimized for both memory efficiency and speed
- **Throughput:** ~2-3x faster than memory-constrained setup

### Evaluation System (CPU-only + 30GB RAM)
- **CPU Memory:** ~20-28GB (under 30GB limit)
- **Inference Speed:** CPU-bound, suitable for evaluation workloads
- **Compatibility:** Runs on Windows without CUDA dependencies

### Legacy System (A100 + 30GB RAM)  
- **GPU VRAM:** ~35-38GB (under 39GB limit)
- **CPU Memory:** ~25-28GB (under 30GB limit)
- **Training Speed:** Memory-optimized over speed

## Automatic CPU Compatibility

The system automatically detects CPU-only environments and adjusts settings:
- **FP16 → FP32**: FP16 training disabled on CPU
- **8-bit optimizers → AdamW**: bitsandbytes requires CUDA
- **DeepSpeed → disabled**: ZeRO stages disabled on CPU
- **Accelerate mapping → balanced**: Device mapping adjusted for CPU

## Troubleshooting

### Training System Issues
- **GPU OOM**: Reduce `micro_batch_per_gpu` from 4 to 2
- **CPU OOM**: Reduce `reduce_bucket_size` and `allgather_bucket_size`
- **DeepSpeed errors**: Ensure DeepSpeed >=0.9.0 with CUDA support

### Evaluation System Issues  
- **CPU OOM**: Reduce `micro_batch_per_gpu` to 1, disable checkpointing
- **Windows compatibility**: Use `cpu_only_30gb_ram.json` configuration
- **Import errors**: Ensure PyTorch CPU-only version is installed

### Common Issues
- **CUDA not found**: System automatically switches to CPU mode
- **Memory fragmentation**: Restart Python process between runs
- **Performance**: CPU evaluation is significantly slower than GPU training

### Customization

You can adjust the following parameters based on your specific needs:
- `micro_batch_per_gpu`: Increase/decrease based on model size
- `gradient_accumulation_steps`: Adjust for different effective batch sizes
- `reduce_bucket_size`: Smaller values = lower memory, slower communication
- `lr`: Learning rate can be tuned based on your training objectives

### Performance Tips
1. Use pinned memory for faster CPU-GPU transfers
2. Enable overlap communication for better throughput
3. Monitor GPU memory usage and adjust micro-batch size accordingly
4. Consider reducing `number_checkpoints` if CPU memory is limited