"""
Component registry for dynamic loading in TRIDENT-Net.

Maps YAML class paths to Python classes and handles component instantiation.

Author: Yağızhan Keskin
"""

import importlib
from typing import Dict, Any, Type, Optional
import logging


class ComponentRegistry:
    """
    Registry for TRIDENT-Net components.
    
    Maps class paths from tasks.yml to actual Python classes.
    """
    
    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._register_builtin_components()
    
    def _register_builtin_components(self):
        """Register all built-in TRIDENT-Net components."""
        
        # TRIDENT-I components
        component_mappings = {
            # TRIDENT-I branch (legacy + new)
            "trident_i.Frag3D": "trident.trident_i.frag3d.Frag3D",
            "trident_i.FlashNetV": "trident.trident_i.flashnet_v.FlashNetV", 
            "trident_i.DualVision": "trident.trident_i.dualvision.DualVision",
            
            # NEW I-branch modules
            "trident_i.videox3d.VideoFrag3Dv2": "trident.trident_i.videox3d.VideoFrag3Dv2",
            "trident_i.dualvision_v2.DualVisionV2": "trident.trident_i.dualvision_v2.DualVisionV2",
            
            # TRIDENT-T branch (legacy + new)
            "trident_t.PlumeDetLite": "trident.trident_t.plumedet_lite.PlumeDetLite",
            "trident_t.CoolCurve3": "trident.trident_t.coolcurve3.CoolCurve3",
            
            # NEW T-branch modules
            "trident_t.ir_dettrack_v2.PlumeDetXL": "trident.trident_t.ir_dettrack_v2.PlumeDetXL",
            
            # TRIDENT-T branch (legacy + existing)
            "trident_t.coolcurve3.CoolCurve3": "trident.trident_t.coolcurve3.CoolCurve3",
            
            # TRIDENT-R branch
            "trident_r.kinefeat.KineFeat": "trident.trident_r.kinefeat.KineFeat",
            "trident_r.geomlp.GeoMLP": "trident.trident_r.geomlp.GeoMLP", 
            "trident_r.tiny_temporal_former.TinyTempoFormer": "trident.trident_r.tiny_temporal_former.TinyTempoFormer",
            
            # Fusion & Guard
            "fusion_guard.cross_attn_fusion.CrossAttnFusion": "trident.fusion_guard.cross_attn_fusion.CrossAttnFusion",
            "fusion_guard.calib_glm.CalibGLM": "trident.fusion_guard.calib_glm.CalibGLM",
            "fusion_guard.spoof_shield.SpoofShield": "trident.fusion_guard.spoof_shield.SpoofShield",
            
            # Data components
            "data.dataset.VideoJsonlDataset": "trident.data.dataset.VideoJsonlDataset",
            "data.transforms.AlbuStereoClip": "trident.data.transforms.AlbuStereoClip",
            "data.video_ring.VideoRing": "trident.data.video_ring.VideoRing",
            "data.collate.pad_tracks_collate": "trident.data.collate.pad_tracks_collate",
            "data.synthetic.SyntheticVideoJsonl": "trident.data.synthetic.SyntheticVideoJsonl",
            
            # XAI
            "xai_text.templater.Templater": "trident.xai_text.templater.Templater",
            "xai_text.small_llm_reporter.SmallLLMReporter": "trident.xai_text.small_llm_reporter.SmallLLMReporter",
            
            # Analytics (optional)
            "analytics.ClusterMiner": "trident.analytics.cluster_miner.ClusterMiner",
            "analytics.RuleMiner": "trident.analytics.rule_miner.RuleMiner",
        }
        
        for yaml_path, python_path in component_mappings.items():
            try:
                cls = self._import_class(python_path)
                self._registry[yaml_path] = cls
                logging.debug(f"Registered component: {yaml_path} -> {python_path}")
            except ImportError as e:
                logging.warning(f"Could not register {yaml_path}: {e}")
            except Exception as e:
                logging.error(f"Error registering {yaml_path}: {e}")
    
    def _import_class(self, class_path: str) -> Type:
        """
        Import a class from a module path.
        
        Args:
            class_path: Full path like "module.submodule.ClassName"
            
        Returns:
            The imported class
        """
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def register(self, yaml_path: str, python_class: Type):
        """
        Register a component class.
        
        Args:
            yaml_path: Path used in YAML (e.g., "trident_i.Frag3D")
            python_class: The actual Python class
        """
        self._registry[yaml_path] = python_class
        logging.info(f"Registered component: {yaml_path}")
    
    def get_class(self, yaml_path: str) -> Optional[Type]:
        """
        Get a component class by its YAML path.
        
        Args:
            yaml_path: Path from tasks.yml
            
        Returns:
            The component class or None if not found
        """
        if yaml_path in self._registry:
            return self._registry[yaml_path]
        
        # Try to import dynamically if not in registry
        try:
            # Convert YAML path to Python path (basic heuristic)
            if '.' in yaml_path:
                module_part, class_part = yaml_path.split('.', 1)
                python_path = f"trident.{module_part}.{class_part.lower()}.{class_part}"
                cls = self._import_class(python_path)
                self._registry[yaml_path] = cls
                return cls
        except Exception as e:
            logging.error(f"Could not import {yaml_path}: {e}")
        
        return None
    
    def create_component(self, yaml_path: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a component instance.
        
        Args:
            yaml_path: Path from tasks.yml
            config: Optional configuration dictionary
            
        Returns:
            Component instance
        """
        cls = self.get_class(yaml_path)
        if cls is None:
            raise ValueError(f"Component not found: {yaml_path}")
        
        if config is None:
            config = {}
        
        try:
            # Try to instantiate with config
            if hasattr(cls, '__init__'):
                import inspect
                sig = inspect.signature(cls.__init__)
                
                # Filter config to only include parameters the class accepts
                valid_params = set(sig.parameters.keys()) - {'self'}
                filtered_config = {k: v for k, v in config.items() if k in valid_params}
                
                return cls(**filtered_config)
            else:
                return cls()
                
        except Exception as e:
            logging.error(f"Error creating {yaml_path}: {e}")
            raise
    
    def list_components(self) -> Dict[str, str]:
        """List all registered components."""
        return {k: str(v) for k, v in self._registry.items()}


# Global registry instance
_global_registry = ComponentRegistry()


def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry


def register_component(yaml_path: str, python_class: Type):
    """Register a component in the global registry."""
    _global_registry.register(yaml_path, python_class)


def create_component(yaml_path: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create a component using the global registry."""
    return _global_registry.create_component(yaml_path, config)


def get_component_class(yaml_path: str) -> Optional[Type]:
    """Get a component class using the global registry."""
    return _global_registry.get_class(yaml_path)


def create_component_from_config(component_config: Dict[str, Any]) -> Any:
    """
    Create a component from a complete config dictionary.
    
    Expected format:
    {
        "class": "trident_i.Frag3D",
        "config": {"param1": "value1", ...}
    }
    """
    yaml_path = component_config.get("class")
    if not yaml_path:
        raise ValueError("Component config missing 'class' field")
    
    config = component_config.get("config", {})
    return create_component(yaml_path, config)