# TRIDENT-Net Recommended Settings for A100 39GB + 30GB CPU

This directory contains optimized DeepSpeed configurations for specific hardware setups.

## A100 39GB + 30GB CPU Configuration

**File:** `a100_39gb_30gb_cpu.json`

### Hardware Requirements
- **GPU:** NVIDIA A100 40GB (39GB usable VRAM)
- **CPU Memory:** 30GB available for offloading
- **System:** Single GPU setup

### Configuration Details

#### Memory Optimization
- **Micro-batch size:** 2 (per GPU)
- **Gradient accumulation:** 4 steps
- **Effective batch size:** 8 (2 × 4)
- **ZeRO Stage:** 2 with CPU offload for optimizer and parameters
- **Mixed Precision:** FP16 enabled

#### DeepSpeed ZeRO Settings
- **Optimizer offload:** CPU with pinned memory
- **Parameter offload:** CPU with pinned memory  
- **Reduced bucket sizes:** 200MB for better memory efficiency
- **Activation checkpointing:** Enabled with CPU checkpointing

#### Usage

**Via CLI:**
```bash
python -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --deepspeed-config configs/a100_39gb_30gb_cpu.json \
  --use-fp16 \
  --grad-accum-steps 4 \
  --max-gpu-mem 39GiB \
  --cpu-mem 30GiB \
  --zero-stage 2
```

**Via DeepSpeed launcher:**
```bash
deepspeed --num_gpus 1 -m trident.runtime.memory_efficient_cli \
  --config tasks.yml \
  --deepspeed-config configs/a100_39gb_30gb_cpu.json \
  --use-fp16 \
  --synthetic
```

### Expected Memory Usage
- **GPU VRAM:** ~35-38GB (under 39GB limit)
- **CPU Memory:** ~25-28GB (under 30GB limit)
- **Training Speed:** Optimized for memory efficiency over speed

### Troubleshooting

If you encounter `DeepSpeedCPUAdam` compatibility issues:
1. Ensure DeepSpeed is properly installed: `pip install deepspeed>=0.9.0`
2. Verify CPU has sufficient memory (30GB+)
3. Check that the configuration uses `"type": "DeepSpeedCPUAdam"`

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