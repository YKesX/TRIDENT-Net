"""
Configuration loading and validation for TRIDENT-Net.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import yaml

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Simple fallback for pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    PYDANTIC_AVAILABLE = False

import torch


class PathConfig(BaseModel):
    """Path configuration."""
    data_root: str = "./data"
    runs_root: str = "./runs"  
    ckpt_root: str = "./checkpoints"


class ComponentConfig(BaseModel):
    """Individual component configuration."""
    class_path: str = Field(..., alias="class")
    kind: str
    inputs: List[str]
    outputs: List[str]
    loss: str
    metrics: List[str]
    config: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    """Dataset configuration."""
    type: str
    path: str


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    name: str = "adamw"
    lr: float = 3e-4
    wd: float = 0.01
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


class TaskConfig(BaseModel):
    """Task configuration."""
    run: str
    component: Optional[str] = None
    components: Optional[List[str]] = None
    dataset: str
    val: Optional[str] = None
    freeze: Optional[List[str]] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    epochs: int = 50
    batch_size: int = 32
    save_to: str
    
    # Additional fields for specific task types
    checkpoint_map: Optional[Dict[str, str]] = None
    features_from: Optional[List[str]] = None
    inputs_from: Optional[List[str]] = None
    rules_source: Optional[str] = None
    metrics: Optional[List[str]] = None
    report_to: Optional[str] = None
    graph: Optional[Dict[str, Any]] = None


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    steps: List[str]


class TridentConfig(BaseModel):
    """Main TRIDENT configuration."""
    version: int = 1
    name: str = "trident-net"
    python: str = ">=3.10"
    framework: str = "pytorch"
    
    paths: PathConfig = Field(default_factory=PathConfig)
    components: Dict[str, ComponentConfig] = Field(default_factory=dict)
    datasets: Dict[str, DatasetConfig] = Field(default_factory=dict)
    tasks: Dict[str, TaskConfig] = Field(default_factory=dict)
    pipelines: Dict[str, PipelineConfig] = Field(default_factory=dict)
    
    @validator('components', pre=True)
    def validate_components(cls, v):
        """Ensure component classes use proper naming."""
        for name, comp in v.items():
            if isinstance(comp, dict):
                # Handle 'class' field alias
                if 'class' in comp:
                    comp['class_path'] = comp.pop('class')
        return v


class ConfigLoader:
    """Load and validate TRIDENT configuration files."""
    
    def __init__(self):
        self.config: Optional[TridentConfig] = None
        self.raw_config: Dict[str, Any] = {}
    
    def load_config(self, config_path: Union[str, Path]) -> TridentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Store raw config for access to fields not in pydantic model
        self.raw_config = config_data.copy()
        
        # Resolve path variables
        config_data = self._resolve_path_variables(config_data)
        
        # Validate and create config object
        self.config = TridentConfig(**config_data)
        
        return self.config
    
    def _resolve_path_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve ${paths.xxx} variables in configuration."""
        paths = config_data.get('paths', {})
        
        def resolve_recursive(obj):
            if isinstance(obj, dict):
                return {k: resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_recursive(item) for item in obj]
            elif isinstance(obj, str):
                # Replace path variables
                for path_key, path_value in paths.items():
                    placeholder = f"${{paths.{path_key}}}"
                    if placeholder in obj:
                        obj = obj.replace(placeholder, path_value)
                return obj
            else:
                return obj
        
        return resolve_recursive(config_data)
    
    def get_component_config(self, component_name: str) -> ComponentConfig:
        """Get configuration for a specific component."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if component_name not in self.config.components:
            raise KeyError(f"Component '{component_name}' not found in configuration")
        
        return self.config.components[component_name]
    
    def get_task_config(self, task_name: str) -> TaskConfig:
        """Get configuration for a specific task."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if task_name not in self.config.tasks:
            raise KeyError(f"Task '{task_name}' not found in configuration")
        
        return self.config.tasks[task_name]
    
    def get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a specific dataset."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if dataset_name not in self.config.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration")
        
        return self.config.datasets[dataset_name]
    
    def get_pipeline_config(self, pipeline_name: str) -> PipelineConfig:
        """Get configuration for a specific pipeline."""
        if self.config is None:
            raise RuntimeError("No configuration loaded")
        
        if pipeline_name not in self.config.pipelines:
            raise KeyError(f"Pipeline '{pipeline_name}' not found in configuration")
        
        return self.config.pipelines[pipeline_name]
    
    def create_optimizer(self, config: OptimizerConfig, parameters) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        optimizer_map = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': torch.optim.SGD,
            'rmsprop': torch.optim.RMSprop,
        }
        
        optimizer_class = optimizer_map.get(config.name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {config.name}")
        
        # Build optimizer kwargs
        kwargs = {
            'lr': config.lr,
        }
        
        if config.name.lower() in ['adam', 'adamw']:
            kwargs.update({
                'betas': config.betas,
                'eps': config.eps,
                'weight_decay': config.wd,
            })
        elif config.name.lower() == 'sgd':
            kwargs.update({
                'momentum': config.betas[0] if config.betas else 0.9,
                'weight_decay': config.wd,
            })
        elif config.name.lower() == 'rmsprop':
            kwargs.update({
                'eps': config.eps,
                'weight_decay': config.wd,
            })
        
        return optimizer_class(parameters, **kwargs)
    
    def get_component_class(self, component_name: str):
        """Get the actual component class from configuration."""
        component_config = self.get_component_config(component_name)
        class_path = component_config.class_path
        
        # Parse class path like "trident_i.FragCNN"
        if '.' in class_path:
            module_path, class_name = class_path.rsplit('.', 1)
        else:
            module_path = class_path
            class_name = class_path
        
        # Map module paths to actual imports
        module_map = {
            'trident_i': 'trident.i_models',
            'trident_r': 'trident.r_models', 
            'trident_t': 'trident.t_models',
            'fusion_guard': 'trident.fusion_guard',
        }
        
        actual_module = module_map.get(module_path, module_path)
        
        try:
            import importlib
            module = importlib.import_module(actual_module)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import {class_path}: {e}")
    
    def create_component(self, component_name: str):
        """Create component instance from configuration."""
        component_config = self.get_component_config(component_name)
        component_class = self.get_component_class(component_name)
        
        # Create component with config
        return component_class(**component_config.config)
    
    def validate_task_dependencies(self, task_name: str) -> List[str]:
        """Validate task dependencies and return required components."""
        task_config = self.get_task_config(task_name)
        required_components = []
        
        if task_config.component:
            required_components.append(task_config.component)
        
        if task_config.components:
            required_components.extend(task_config.components)
        
        if task_config.freeze:
            required_components.extend(task_config.freeze)
        
        if task_config.features_from:
            required_components.extend(task_config.features_from)
        
        if task_config.inputs_from:
            required_components.extend(task_config.inputs_from)
        
        # Check that all required components exist
        missing_components = []
        for comp_name in required_components:
            if comp_name not in self.config.components:
                missing_components.append(comp_name)
        
        if missing_components:
            raise ValueError(f"Task '{task_name}' requires missing components: {missing_components}")
        
        return list(set(required_components))  # Remove duplicates


# Global config loader instance
config_loader = ConfigLoader()


def load_config(config_path: Union[str, Path]) -> TridentConfig:
    """Load TRIDENT configuration."""
    return config_loader.load_config(config_path)


def get_config() -> Optional[TridentConfig]:
    """Get currently loaded configuration."""
    return config_loader.config