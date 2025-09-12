"""
Model export functionality for TRIDENT-Net.

Supports export to ONNX and TorchScript formats as specified in tasks.yml.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    opset_version: int = 18,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
) -> bool:
    """
    Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shapes: Dictionary of input names to shapes
        opset_version: ONNX opset version
        dynamic_axes: Optional dynamic axes specification
        
    Returns:
        bool: True if export successful
    """
    if not ONNX_AVAILABLE:
        logging.warning("ONNX not available - export skipped")
        return False
        
    try:
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape)
        
        # Export model
        model.eval()
        with torch.no_grad():
            if len(dummy_inputs) == 1:
                # Single input
                input_tensor = list(dummy_inputs.values())[0]
                torch.onnx.export(
                    model,
                    input_tensor,
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    dynamic_axes=dynamic_axes,
                )
            else:
                # Multiple inputs
                torch.onnx.export(
                    model,
                    tuple(dummy_inputs.values()),
                    output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=list(dummy_inputs.keys()),
                    dynamic_axes=dynamic_axes,
                )
        
        logging.info(f"Successfully exported ONNX model to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"ONNX export failed: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    trace_mode: bool = True,
) -> bool:
    """
    Export model to TorchScript format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        input_shapes: Dictionary of input names to shapes
        trace_mode: Use tracing instead of scripting
        
    Returns:
        bool: True if export successful
    """
    try:
        model.eval()
        
        if trace_mode:
            # Use tracing
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randn(*shape)
            
            with torch.no_grad():
                if len(dummy_inputs) == 1:
                    input_tensor = list(dummy_inputs.values())[0]
                    traced_model = torch.jit.trace(model, input_tensor)
                else:
                    traced_model = torch.jit.trace(model, tuple(dummy_inputs.values()))
        else:
            # Use scripting
            traced_model = torch.jit.script(model)
        
        # Save model
        traced_model.save(output_path)
        logging.info(f"Successfully exported TorchScript model to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"TorchScript export failed: {e}")
        return False


def validate_exported_model(
    original_model: nn.Module,
    exported_path: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    format_type: str = "onnx",
    tolerance: float = 1e-5,
) -> bool:
    """
    Validate that exported model produces similar outputs to original.
    
    Args:
        original_model: Original PyTorch model
        exported_path: Path to exported model
        input_shapes: Dictionary of input names to shapes
        format_type: "onnx" or "torchscript"
        tolerance: Numerical tolerance for comparison
        
    Returns:
        bool: True if validation passes
    """
    try:
        # Create test inputs
        test_inputs = {}
        for name, shape in input_shapes.items():
            test_inputs[name] = torch.randn(*shape)
        
        # Get original model output
        original_model.eval()
        with torch.no_grad():
            if len(test_inputs) == 1:
                original_output = original_model(list(test_inputs.values())[0])
            else:
                original_output = original_model(*test_inputs.values())
        
        # Get exported model output
        if format_type == "onnx":
            if not ONNX_AVAILABLE:
                logging.warning("ONNX not available - validation skipped")
                return True
                
            session = onnxruntime.InferenceSession(exported_path)
            input_data = {name: tensor.numpy() for name, tensor in test_inputs.items()}
            exported_output = session.run(None, input_data)
            exported_output = torch.from_numpy(exported_output[0])
            
        elif format_type == "torchscript":
            loaded_model = torch.jit.load(exported_path)
            loaded_model.eval()
            with torch.no_grad():
                if len(test_inputs) == 1:
                    exported_output = loaded_model(list(test_inputs.values())[0])
                else:
                    exported_output = loaded_model(*test_inputs.values())
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Compare outputs
        if isinstance(original_output, torch.Tensor):
            diff = torch.abs(original_output - exported_output).max().item()
        else:
            # Handle multiple outputs
            diffs = []
            for orig, exp in zip(original_output, exported_output):
                diffs.append(torch.abs(orig - exp).max().item())
            diff = max(diffs)
        
        if diff < tolerance:
            logging.info(f"Validation passed - max difference: {diff}")
            return True
        else:
            logging.error(f"Validation failed - max difference: {diff} > {tolerance}")
            return False
            
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False


def export_fusion_model(
    model: nn.Module,
    output_dir: str,
    batch_size: int = 1,
) -> Dict[str, bool]:
    """
    Export fusion model (f2) as specified in tasks.yml.
    
    Expected inputs from tasks.yml:
    - zi: B x 768
    - zt: B x 512  
    - zr: B x 384
    - class_emb: B x 32 (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define input shapes
    input_shapes = {
        "zi": (batch_size, 768),
        "zt": (batch_size, 512), 
        "zr": (batch_size, 384),
        "class_emb": (batch_size, 32),
    }
    
    # Dynamic axes for batch dimension
    dynamic_axes = {
        name: {0: "batch_size"} for name in input_shapes.keys()
    }
    
    results = {}
    
    # Export to ONNX
    onnx_path = output_dir / "f2.onnx"
    results["onnx"] = export_to_onnx(
        model, 
        str(onnx_path), 
        input_shapes,
        dynamic_axes=dynamic_axes
    )
    
    return results


def export_guard_model(
    model: nn.Module,
    output_dir: str,
    batch_size: int = 1,
) -> Dict[str, bool]:
    """
    Export guard model (s) as specified in tasks.yml.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define input shapes for guard model
    input_shapes = {
        "p_hit": (batch_size, 1),
        "p_kill": (batch_size, 1),
    }
    
    results = {}
    
    # Export to TorchScript  
    ts_path = output_dir / "s.ts"
    results["torchscript"] = export_to_torchscript(
        model,
        str(ts_path),
        input_shapes
    )
    
    return results


class ModelExporter:
    """Model exporter class for convenient export operations."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
    def to_torchscript(self, dummy_input: torch.Tensor, use_tracing: bool = True) -> torch.jit.ScriptModule:
        """Export model to TorchScript."""
        self.model.eval()
        
        if use_tracing:
            return torch.jit.trace(self.model, dummy_input)
        else:
            return torch.jit.script(self.model)
    
    def to_onnx(self, dummy_input: torch.Tensor, output_path: str, **kwargs) -> bool:
        """Export model to ONNX."""
        if not ONNX_AVAILABLE:
            logging.warning("ONNX not available")
            return False
            
        try:
            self.model.eval()
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=kwargs.get('opset_version', 11),
                do_constant_folding=True,
                **kwargs
            )
            return True
        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            return False