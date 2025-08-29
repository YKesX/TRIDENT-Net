"""
Execution graph for TRIDENT-Net inference.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import logging

import torch
import torch.nn as nn

from .config import ConfigLoader, TridentConfig
from ..common.types import FeatureVec, OutcomeEstimate, EventToken


@dataclass
class GraphNode:
    """Node in the execution graph."""
    name: str
    component: nn.Module
    inputs: List[str]
    outputs: List[str]
    dependencies: List[str]
    frozen: bool = False


class ExecutionGraph:
    """
    Execution graph for TRIDENT-Net inference pipeline.
    
    Manages component dependencies, execution order, and data flow
    between different modality processors and fusion modules.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.config = config_loader.config
        
        self.nodes: Dict[str, GraphNode] = {}
        self.execution_order: List[str] = []
        self.loaded_checkpoints: Dict[str, str] = {}
        
        # Runtime state
        self.feature_cache: Dict[str, FeatureVec] = {}
        self.event_cache: Dict[str, List[EventToken]] = {}
        self.output_cache: Dict[str, Any] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def build_graph(
        self, 
        components: List[str], 
        frozen_components: Optional[List[str]] = None
    ) -> None:
        """
        Build execution graph from component list.
        
        Args:
            components: List of component names to include
            frozen_components: List of components to freeze
        """
        if frozen_components is None:
            frozen_components = []
        
        self.nodes = {}
        
        # Create nodes for each component
        for comp_name in components:
            comp_config = self.config_loader.get_component_config(comp_name)
            component = self.config_loader.create_component(comp_name)
            
            # Determine dependencies from inputs
            dependencies = self._get_dependencies(comp_config.inputs)
            
            node = GraphNode(
                name=comp_name,
                component=component,
                inputs=comp_config.inputs,
                outputs=comp_config.outputs,
                dependencies=dependencies,
                frozen=comp_name in frozen_components,
            )
            
            self.nodes[comp_name] = node
        
        # Compute execution order
        self.execution_order = self._topological_sort()
        
        self.logger.info(f"Built execution graph with {len(self.nodes)} nodes")
        self.logger.info(f"Execution order: {' -> '.join(self.execution_order)}")
    
    def _get_dependencies(self, inputs: List[str]) -> List[str]:
        """Determine component dependencies from input specifications."""
        dependencies = []
        
        # Parse input specifications to find dependencies
        for input_spec in inputs:
            # Format: "modality_feature: BxD" or "input_name: shape"
            if ':' in input_spec:
                input_name = input_spec.split(':')[0].strip()
                
                # Map input names to component dependencies
                if input_name.startswith('z'):
                    # Feature vectors from other components
                    if input_name == 'zr':
                        dependencies.extend(['r1', 'r2', 'r3'])
                    elif input_name == 'zi':
                        dependencies.extend(['i1', 'i2', 'i3'])
                    elif input_name == 'zt':
                        dependencies.extend(['t1', 't2'])
        
        return dependencies
    
    def _topological_sort(self) -> List[str]:
        """Compute topological ordering of graph nodes."""
        # Kahn's algorithm
        in_degree = {name: 0 for name in self.nodes}
        
        # Count incoming edges
        for node_name, node in self.nodes.items():
            for dep in node.dependencies:
                if dep in in_degree:
                    in_degree[node_name] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current node
            for node_name, node in self.nodes.items():
                if current in node.dependencies:
                    in_degree[node_name] -= 1
                    if in_degree[node_name] == 0:
                        queue.append(node_name)
        
        if len(result) != len(self.nodes):
            raise ValueError("Circular dependency detected in execution graph")
        
        return result
    
    def load_checkpoints(self, checkpoint_map: Dict[str, str]) -> None:
        """
        Load model checkpoints for components.
        
        Args:
            checkpoint_map: Mapping of component names to checkpoint paths
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for comp_name, ckpt_path in checkpoint_map.items():
            if comp_name not in self.nodes:
                self.logger.warning(f"Component {comp_name} not in graph, skipping checkpoint")
                continue
            
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                self.logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue
            
            try:
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location=device)
                
                # Load model state
                if 'model_state_dict' in checkpoint:
                    self.nodes[comp_name].component.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.nodes[comp_name].component.load_state_dict(checkpoint)
                
                # Move to device
                self.nodes[comp_name].component.to(device)
                
                # Set to eval mode if frozen
                if self.nodes[comp_name].frozen:
                    self.nodes[comp_name].component.eval()
                
                self.loaded_checkpoints[comp_name] = str(ckpt_path)
                self.logger.info(f"Loaded checkpoint for {comp_name}: {ckpt_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint for {comp_name}: {e}")
                raise
    
    def freeze_components(self, component_names: List[str]) -> None:
        """Freeze specified components."""
        for comp_name in component_names:
            if comp_name in self.nodes:
                self.nodes[comp_name].frozen = True
                self.nodes[comp_name].component.eval()
                
                # Freeze parameters
                for param in self.nodes[comp_name].component.parameters():
                    param.requires_grad = False
                
                self.logger.info(f"Frozen component: {comp_name}")
    
    def execute(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute the graph with given inputs.
        
        Args:
            inputs: Input tensors keyed by modality names
            
        Returns:
            Dictionary of outputs from all components
        """
        # Clear caches
        self.feature_cache = {}
        self.event_cache = {}
        self.output_cache = {}
        
        # Execute components in topological order
        for comp_name in self.execution_order:
            node = self.nodes[comp_name]
            
            try:
                # Prepare inputs for this component
                comp_inputs = self._prepare_component_inputs(node, inputs)
                
                # Skip if inputs not available
                if not comp_inputs:
                    self.logger.warning(f"No inputs available for {comp_name}, skipping")
                    continue
                
                # Execute component
                if node.frozen:
                    with torch.no_grad():
                        output = node.component(**comp_inputs)
                else:
                    output = node.component(**comp_inputs)
                
                # Store outputs
                self._store_component_outputs(comp_name, output)
                
                self.logger.debug(f"Executed component: {comp_name}")
                
            except Exception as e:
                self.logger.error(f"Error executing component {comp_name}: {e}")
                raise
        
        return self.output_cache
    
    def _prepare_component_inputs(
        self, 
        node: GraphNode, 
        raw_inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Prepare inputs for a component based on its input specification."""
        comp_inputs = {}
        
        for input_spec in node.inputs:
            if ':' in input_spec:
                input_name = input_spec.split(':')[0].strip()
                
                # Handle different input types
                if input_name in raw_inputs:
                    # Direct input from raw data
                    comp_inputs[input_name] = raw_inputs[input_name]
                elif input_name in ['zr', 'zi', 'zt']:
                    # Feature vectors from cache
                    if input_name in self.feature_cache:
                        comp_inputs[input_name] = self.feature_cache[input_name]
                elif input_name == 'events':
                    # Event tokens from cache
                    all_events = []
                    for events in self.event_cache.values():
                        all_events.extend(events)
                    comp_inputs[input_name] = all_events
                elif input_name in ['geom', 'priors']:
                    # Metadata inputs
                    if input_name in raw_inputs:
                        comp_inputs[input_name] = raw_inputs[input_name]
                elif input_name.startswith('p_') and input_name in self.output_cache:
                    # Probability inputs from previous fusion modules
                    comp_inputs[input_name] = self.output_cache[input_name]
        
        return comp_inputs
    
    def _store_component_outputs(self, comp_name: str, output: Any) -> None:
        """Store component outputs in appropriate caches."""
        
        if isinstance(output, tuple) and len(output) == 2:
            # Branch module output: (FeatureVec, List[EventToken])
            feature_vec, events = output
            
            if isinstance(feature_vec, FeatureVec):
                # Map component to feature cache key
                if comp_name.startswith('r'):
                    self.feature_cache['zr'] = feature_vec
                elif comp_name.startswith('i'):
                    self.feature_cache['zi'] = feature_vec
                elif comp_name.startswith('t'):
                    self.feature_cache['zt'] = feature_vec
                
                self.event_cache[comp_name] = events
        
        elif isinstance(output, OutcomeEstimate):
            # Fusion/guard module output
            self.output_cache[f'{comp_name}_outcome'] = output
            self.output_cache[f'p_{comp_name}'] = output.p_outcome
            self.output_cache[f'binary_{comp_name}'] = output.binary_outcome
            
            if output.explanation:
                self.output_cache[f'{comp_name}_explanation'] = output.explanation
        
        elif isinstance(output, dict):
            # Dictionary output - store all keys
            for key, value in output.items():
                self.output_cache[f'{comp_name}_{key}'] = value
        
        else:
            # Generic output
            self.output_cache[comp_name] = output
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of graph execution."""
        return {
            "nodes": len(self.nodes),
            "execution_order": self.execution_order,
            "loaded_checkpoints": list(self.loaded_checkpoints.keys()),
            "frozen_components": [name for name, node in self.nodes.items() if node.frozen],
            "feature_cache_keys": list(self.feature_cache.keys()),
            "event_cache_keys": list(self.event_cache.keys()),
            "output_cache_keys": list(self.output_cache.keys()),
        }
    
    def reset(self) -> None:
        """Reset graph state and caches."""
        self.feature_cache = {}
        self.event_cache = {}
        self.output_cache = {}
        
        self.logger.info("Reset execution graph state")
    
    def to_device(self, device: torch.device) -> None:
        """Move all components to specified device."""
        for node in self.nodes.values():
            node.component.to(device)
        
        self.logger.info(f"Moved graph components to device: {device}")


def create_inference_graph(
    config: TridentConfig,
    components: List[str],
    checkpoint_map: Dict[str, str],
    frozen_components: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> ExecutionGraph:
    """
    Create and configure inference execution graph.
    
    Args:
        config: TRIDENT configuration
        components: List of components to include
        checkpoint_map: Component checkpoint paths
        frozen_components: Components to freeze
        device: Target device
        
    Returns:
        Configured execution graph
    """
    from .config import ConfigLoader
    
    # Create config loader
    config_loader = ConfigLoader()
    config_loader.config = config
    
    # Create graph
    graph = ExecutionGraph(config_loader)
    graph.build_graph(components, frozen_components)
    
    # Load checkpoints
    if checkpoint_map:
        graph.load_checkpoints(checkpoint_map)
    
    # Freeze components
    if frozen_components:
        graph.freeze_components(frozen_components)
    
    # Move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph.to_device(device)
    
    return graph