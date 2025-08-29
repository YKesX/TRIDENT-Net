"""
Inference server for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional
import logging
import asyncio
import json
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .graph import ExecutionGraph
from ..common.utils import move_to_device
from ..data.synthetic import generate_synthetic_batch


# Request/Response models
class InferenceRequest(BaseModel):
    """Request model for inference."""
    data: Dict[str, List[float]]  # Flattened tensor data
    shapes: Dict[str, List[int]]  # Tensor shapes
    metadata: Optional[Dict[str, Any]] = None


class InferenceResponse(BaseModel):
    """Response model for inference."""
    p_outcome: float
    binary_outcome: int
    uncertainty: Optional[float] = None
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    components_loaded: List[str]
    device: str
    version: str


class TridentServer:
    """
    TRIDENT-Net inference server.
    
    Provides REST API endpoints for real-time inference
    using the execution graph.
    """
    
    def __init__(
        self,
        graph: ExecutionGraph,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_cors: bool = True,
    ):
        self.graph = graph
        self.host = host
        self.port = port
        
        # Setup FastAPI
        self.app = FastAPI(
            title="TRIDENT-Net Inference Server",
            description="Multimodal sensor fusion inference API",
            version="0.1.0",
        )
        
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Setup routes
        self._setup_routes()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.request_count = 0
        self.total_processing_time = 0.0
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                components_loaded=list(self.graph.nodes.keys()),
                device=str(next(iter(self.graph.nodes.values())).component.training),  # Approximation
                version="0.1.0",
            )
        
        @self.app.post("/infer", response_model=InferenceResponse)
        async def infer(request: InferenceRequest):
            """Main inference endpoint."""
            import time
            start_time = time.time()
            
            try:
                # Convert request data to tensors
                input_tensors = self._request_to_tensors(request)
                
                # Run inference
                result = await self._run_inference(input_tensors)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000
                
                # Update statistics
                self.request_count += 1
                self.total_processing_time += processing_time
                
                # Extract outcome from result
                outcome_estimate = None
                for key, value in result.items():
                    if "outcome" in key and hasattr(value, 'p_outcome'):
                        outcome_estimate = value
                        break
                
                if outcome_estimate is None:
                    raise HTTPException(status_code=500, detail="No outcome estimate generated")
                
                # Prepare response
                p_outcome = outcome_estimate.p_outcome
                if hasattr(p_outcome, 'item'):
                    p_outcome = p_outcome.item()
                else:
                    p_outcome = float(p_outcome)
                
                binary_outcome = outcome_estimate.binary_outcome
                if hasattr(binary_outcome, 'item'):
                    binary_outcome = binary_outcome.item()
                else:
                    binary_outcome = int(binary_outcome)
                
                uncertainty = None
                if outcome_estimate.uncertainty is not None:
                    if hasattr(outcome_estimate.uncertainty, 'item'):
                        uncertainty = outcome_estimate.uncertainty.item()
                    else:
                        uncertainty = float(outcome_estimate.uncertainty)
                
                return InferenceResponse(
                    p_outcome=p_outcome,
                    binary_outcome=binary_outcome,
                    uncertainty=uncertainty,
                    explanation=outcome_estimate.explanation,
                    processing_time_ms=processing_time,
                )
                
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/infer_batch")
        async def infer_batch(requests: List[InferenceRequest]):
            """Batch inference endpoint."""
            results = []
            
            for request in requests:
                try:
                    result = await infer(request)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch item failed: {e}")
                    results.append({
                        "error": str(e),
                        "p_outcome": 0.0,
                        "binary_outcome": 0,
                        "processing_time_ms": 0.0,
                    })
            
            return results
        
        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            avg_processing_time = (
                self.total_processing_time / max(self.request_count, 1)
            )
            
            return {
                "request_count": self.request_count,
                "avg_processing_time_ms": avg_processing_time,
                "components": list(self.graph.nodes.keys()),
                "device": str(next(iter(self.graph.nodes.values())).component.training),
            }
        
        @self.app.post("/generate_sample")
        async def generate_sample():
            """Generate synthetic sample for testing."""
            try:
                # Generate synthetic batch
                batch = generate_synthetic_batch(batch_size=1)
                
                # Convert to request format
                request_data = {}
                shapes = {}
                
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        request_data[key] = value.flatten().tolist()
                        shapes[key] = list(value.shape)
                
                return {
                    "data": request_data,
                    "shapes": shapes,
                    "metadata": {"synthetic": True},
                }
                
            except Exception as e:
                self.logger.error(f"Sample generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/components")
        async def get_components():
            """Get information about loaded components."""
            component_info = {}
            
            for name, node in self.graph.nodes.items():
                component_info[name] = {
                    "class": node.component.__class__.__name__,
                    "frozen": node.frozen,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                    "dependencies": node.dependencies,
                    "parameters": sum(p.numel() for p in node.component.parameters()),
                }
            
            return {
                "execution_order": self.graph.execution_order,
                "components": component_info,
            }
    
    def _request_to_tensors(self, request: InferenceRequest) -> Dict[str, torch.Tensor]:
        """Convert request data back to tensors."""
        tensors = {}
        
        for key, flat_data in request.data.items():
            if key in request.shapes:
                shape = request.shapes[key]
                # Convert to tensor and reshape
                tensor = torch.tensor(flat_data, dtype=torch.float32).reshape(shape)
                tensors[key] = tensor
        
        # Add metadata
        if request.metadata:
            for key, value in request.metadata.items():
                if key not in tensors:
                    tensors[key] = value
        
        return tensors
    
    async def _run_inference(self, input_tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Run inference on input tensors."""
        # Move to device
        device = next(iter(self.graph.nodes.values())).component.device if self.graph.nodes else torch.device("cpu")
        input_tensors = move_to_device(input_tensors, device)
        
        # Reset graph state
        self.graph.reset()
        
        # Execute graph
        with torch.no_grad():
            outputs = self.graph.execute(input_tensors)
        
        return outputs
    
    def start(self):
        """Start the server."""
        self.logger.info(f"Starting TRIDENT server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
    
    async def start_async(self):
        """Start server asynchronously."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()


class TridentClient:
    """
    Client for TRIDENT-Net inference server.
    
    Provides convenient methods for sending inference requests.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        
    async def infer(
        self,
        data: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        """Send inference request."""
        import aiohttp
        
        # Convert tensors to request format
        request_data = {}
        shapes = {}
        
        for key, tensor in data.items():
            if isinstance(tensor, torch.Tensor):
                request_data[key] = tensor.flatten().tolist()
                shapes[key] = list(tensor.shape)
        
        request = InferenceRequest(
            data=request_data,
            shapes=shapes,
            metadata=metadata,
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/infer",
                json=request.dict(),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return InferenceResponse(**result)
                else:
                    raise RuntimeError(f"Request failed: {response.status}")
    
    async def health_check(self) -> HealthResponse:
        """Check server health."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    return HealthResponse(**result)
                else:
                    raise RuntimeError(f"Health check failed: {response.status}")
    
    async def generate_sample(self) -> Dict[str, Any]:
        """Generate synthetic sample from server."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/generate_sample") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise RuntimeError(f"Sample generation failed: {response.status}")


# Standalone server startup
def run_server(
    config_path: str,
    task_name: str = "infer_realtime",
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Run TRIDENT server from configuration."""
    from .config import load_config, ConfigLoader
    from .graph import create_inference_graph
    
    # Load configuration
    trident_config = load_config(config_path)
    config_loader = ConfigLoader()
    config_loader.config = trident_config
    
    # Get task configuration
    task_config = config_loader.get_task_config(task_name)
    
    # Create inference graph
    components = task_config.graph.get("order", [])
    checkpoint_map = task_config.checkpoint_map or {}
    
    graph = create_inference_graph(
        trident_config,
        components,
        checkpoint_map,
        frozen_components=components,
    )
    
    # Start server
    server = TridentServer(graph, host=host, port=port)
    server.start()