# Makefile for TRIDENT-Net Memory-Efficient Training

.PHONY: help install test-smoke test-cpu clean examples deps

# Default target
help:
	@echo "TRIDENT-Net Memory-Efficient Training"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     Install all dependencies"
	@echo "  deps        Install memory optimization dependencies only"
	@echo "  test-cpu    Run CPU smoke test (no GPU required)"
	@echo "  test-smoke  Run full smoke test (requires GPU)"
	@echo "  examples    Show CLI usage examples"
	@echo "  clean       Clean up temporary files"
	@echo ""
	@echo "Memory-efficient training variants:"
	@echo "  make train-deepspeed    Train with DeepSpeed ZeRO-2"
	@echo "  make train-accelerate   Train with HF Accelerate"
	@echo "  make train-qlora        Train with QLoRA optimizations"

# Install all dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov

# Install only memory optimization dependencies
deps:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	pip install bitsandbytes>=0.41.0
	pip install accelerate>=0.20.0
	pip install deepspeed>=0.9.0
	pip install transformers>=4.30.0

# Run CPU smoke test
test-cpu:
	@echo "Running CPU smoke test..."
	python test_cpu_smoke.py
	@echo "CPU test completed. Check cpu_smoke_test_results.json for details."

# Run full smoke test (requires GPU)
test-smoke:
	@echo "Running full smoke test (requires GPU)..."
	python test_smoke.py
	@echo "Smoke test completed. Check smoke_test_results.json for details."

# Show CLI examples
examples:
	@./memory_training_examples.sh

# Clean temporary files
clean:
	rm -rf __pycache__ *.pyc *.pyo
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build dist
	rm -rf ./offload ./test_offload
	rm -f *smoke_test_results.json
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Memory-efficient training targets
train-deepspeed:
	@echo "Training with DeepSpeed ZeRO-2 Offload..."
	deepspeed --num_gpus 1 -m trident.runtime.memory_efficient_cli \
		--config tasks.yml \
		--use-bf16 \
		--checkpoint-every-layer \
		--grad-accum-steps 8 \
		--optimizer adamw8bit \
		--zero-stage 2 \
		--synthetic

train-accelerate:
	@echo "Training with HF Accelerate device mapping..."
	python -m trident.runtime.memory_efficient_cli \
		--config tasks.yml \
		--use-bf16 \
		--checkpoint-every-layer \
		--grad-accum-steps 8 \
		--optimizer paged_adamw8bit \
		--device-map auto \
		--max-gpu-mem 39GiB \
		--cpu-mem 70GiB \
		--zero-stage 0 \
		--synthetic

train-qlora:
	@echo "Training with QLoRA optimizations..."
	python -m trident.runtime.memory_efficient_cli \
		--config tasks.yml \
		--use-bf16 \
		--checkpoint-every-layer \
		--grad-accum-steps 16 \
		--optimizer paged_adamw8bit \
		--qlora \
		--device-map auto \
		--max-gpu-mem 39GiB \
		--synthetic

# Memory usage monitoring
monitor-memory:
	@echo "Monitoring GPU memory usage..."
	@echo "Press Ctrl+C to stop"
	watch -n 1 nvidia-smi

# Check GPU availability
check-gpu:
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Installation verification
verify-install:
	@echo "Verifying installation..."
	@python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
	@python -c "import bitsandbytes; print('✅ bitsandbytes')" || echo "❌ bitsandbytes not available"
	@python -c "import accelerate; print('✅ accelerate')" || echo "❌ accelerate not available" 
	@python -c "import deepspeed; print('✅ deepspeed')" || echo "❌ deepspeed not available"
	@python -c "from trident.runtime.memory_efficient_trainer import MemoryEfficientTrainer; print('✅ MemoryEfficientTrainer')"
	@echo "Installation verification complete."

# Quick start
quickstart: deps verify-install test-cpu
	@echo ""
	@echo "🎉 Quick start complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make examples' to see CLI usage"
	@echo "2. Run 'make test-smoke' if you have a GPU"
	@echo "3. Run 'make train-deepspeed' to start training"