"""
Small LLM reporter interface for TRIDENT-Net.

Provides interface for generating detailed reports using a CPU-quantized LLM.
This runs off the critical path via job queue as specified in tasks.yml.

Author: Yağızhan Keskin
"""

from typing import Dict, List, Any, Optional, Union
import json
import logging
from pathlib import Path
from threading import Thread
from queue import Queue, Empty
import time


class SmallLLMReporter:
    """
    Interface for CPU-quantized small LLM reporter.
    
    Note: This is a stub implementation. In production, this would interface
    with an actual quantized LLM model for report generation.
    """
    
    def __init__(
        self,
        model: str = "tinyllama-1.1b-report",
        quantization: str = "int8-static", 
        max_tokens: int = 256,
        temperature: float = 0.1,
        template_path: Optional[str] = None,
        prompt_schema: Optional[str] = None,
    ):
        self.model = model
        self.quantization = quantization
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load templates and schema
        self.template = self._load_template(template_path)
        self.schema = self._load_schema(prompt_schema)
        
        # Report generation queue (non-blocking)
        self.report_queue = Queue()
        self.result_queue = Queue()
        
        # Start background worker
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logging.info(f"Initialized SmallLLMReporter with model: {model}")
    
    def _load_template(self, template_path: Optional[str]) -> str:
        """Load report template."""
        if template_path and Path(template_path).exists():
            try:
                with open(template_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logging.warning(f"Could not load template: {e}")
        
        # Default template
        return """
# TRIDENT-Net Analysis Report

## Summary
Detection confidence: {confidence}
Primary outcome: {outcome}
Risk assessment: {risk_level}

## Sensor Analysis
{sensor_analysis}

## Events Detected
{events_summary}

## Attention Analysis
{attention_summary}

## Recommendations
{recommendations}

---
Generated at: {timestamp}
Model: {model_info}
        """.strip()
    
    def _load_schema(self, schema_path: Optional[str]) -> Dict[str, Any]:
        """Load prompt schema."""
        if schema_path and Path(schema_path).exists():
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load schema: {e}")
        
        # Default schema
        return {
            "required_fields": [
                "p_hit", "p_kill", "events", "attention_maps"
            ],
            "optional_fields": [
                "spoof_risk", "geom", "class_id"
            ],
            "output_format": "markdown"
        }
    
    def _worker_loop(self):
        """Background worker for processing report requests."""
        while True:
            try:
                # Get request from queue (blocking)
                request = self.report_queue.get(timeout=1.0)
                
                # Generate report
                report = self._generate_report_sync(request)
                
                # Put result in result queue
                self.result_queue.put({
                    "request_id": request.get("request_id"),
                    "report": report,
                    "timestamp": time.time(),
                })
                
                # Mark task as done
                self.report_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error in report worker: {e}")
    
    def _generate_report_sync(self, request_data: Dict[str, Any]) -> str:
        """
        Generate report synchronously (stub implementation).
        
        In production, this would call the actual LLM model.
        """
        try:
            # Extract data
            p_hit = request_data.get("p_hit", 0.0)
            p_kill = request_data.get("p_kill", 0.0)
            events = request_data.get("events", [])
            attention_maps = request_data.get("attention_maps", {})
            spoof_risk = request_data.get("spoof_risk", 0.0)
            
            # Determine primary outcome and confidence
            if p_hit > p_kill:
                outcome = "HIT"
                confidence = p_hit
            else:
                outcome = "KILL"
                confidence = p_kill
            
            # Risk assessment
            if spoof_risk > 0.7:
                risk_level = "HIGH - Possible spoofing detected"
            elif confidence < 0.3:
                risk_level = "HIGH - Low confidence"
            elif confidence < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            # Analyze sensors
            sensor_analysis = self._analyze_sensors(events, attention_maps)
            
            # Summarize events
            events_summary = self._summarize_events(events)
            
            # Analyze attention
            attention_summary = self._analyze_attention(attention_maps)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                confidence, spoof_risk, events
            )
            
            # Fill template
            report = self.template.format(
                confidence=f"{confidence:.3f}",
                outcome=outcome,
                risk_level=risk_level,
                sensor_analysis=sensor_analysis,
                events_summary=events_summary,
                attention_summary=attention_summary,
                recommendations=recommendations,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_info=f"{self.model} ({self.quantization})",
            )
            
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def _analyze_sensors(
        self, 
        events: List[Dict[str, Any]], 
        attention_maps: Dict[str, Any]
    ) -> str:
        """Analyze sensor contributions."""
        if not events:
            return "No significant sensor events detected."
        
        # Group events by sensor type
        sensor_counts = {}
        for event in events:
            sensor_type = event.get("type", "unknown")
            sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1
        
        analysis = []
        for sensor, count in sensor_counts.items():
            analysis.append(f"- {sensor}: {count} events")
        
        return "\n".join(analysis)
    
    def _summarize_events(self, events: List[Dict[str, Any]]) -> str:
        """Summarize detected events."""
        if not events:
            return "No events detected."
        
        # Sort by score
        sorted_events = sorted(
            events, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        summary = []
        for i, event in enumerate(sorted_events[:5]):  # Top 5 events
            event_type = event.get("type", "unknown")
            score = event.get("score", 0)
            timestamp = event.get("t_ms", 0)
            summary.append(f"{i+1}. {event_type} (score: {score:.3f}, t: {timestamp}ms)")
        
        return "\n".join(summary)
    
    def _analyze_attention(self, attention_maps: Dict[str, Any]) -> str:
        """Analyze attention patterns."""
        if not attention_maps:
            return "No attention information available."
        
        analysis = []
        for modality, attn_data in attention_maps.items():
            if isinstance(attn_data, dict):
                max_attn = attn_data.get("max_attention", 0)
                analysis.append(f"- {modality}: peak attention {max_attn:.3f}")
        
        return "\n".join(analysis) if analysis else "Distributed attention pattern."
    
    def _generate_recommendations(
        self,
        confidence: float,
        spoof_risk: float, 
        events: List[Dict[str, Any]]
    ) -> str:
        """Generate actionable recommendations."""
        recommendations = []
        
        if confidence < 0.3:
            recommendations.append("- Consider additional sensor data")
            recommendations.append("- Verify detection with alternative methods")
        
        if spoof_risk > 0.5:
            recommendations.append("- Investigate potential spoofing")
            recommendations.append("- Cross-validate with independent systems")
        
        if not events:
            recommendations.append("- No significant events - monitor for changes")
        
        if not recommendations:
            recommendations.append("- Detection appears reliable")
            recommendations.append("- Continue monitoring")
        
        return "\n".join(recommendations)
    
    def enqueue_report(
        self,
        p_hit: float,
        p_kill: float,
        events: List[Dict[str, Any]],
        attention_maps: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Enqueue a report generation request (non-blocking).
        
        Returns:
            str: Request ID for tracking
        """
        if request_id is None:
            request_id = f"report_{int(time.time() * 1000)}"
        
        request_data = {
            "request_id": request_id,
            "p_hit": p_hit,
            "p_kill": p_kill,
            "events": events,
            "attention_maps": attention_maps or {},
            **kwargs
        }
        
        self.report_queue.put(request_data)
        logging.info(f"Enqueued report request: {request_id}")
        
        return request_id
    
    def get_report(self, request_id: str, timeout: float = 1.0) -> Optional[str]:
        """
        Get a completed report by ID.
        
        Args:
            request_id: Report request ID
            timeout: Timeout in seconds
            
        Returns:
            Report string or None if not ready
        """
        try:
            while True:
                result = self.result_queue.get(timeout=timeout)
                if result["request_id"] == request_id:
                    return result["report"]
                else:
                    # Put back for other requests
                    self.result_queue.put(result)
        except Empty:
            return None
    
    def generate_immediate_report(
        self,
        p_hit: float,
        p_kill: float, 
        events: List[Dict[str, Any]],
        attention_maps: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate report immediately (blocking).
        
        For testing purposes only - production should use async queue.
        """
        request_data = {
            "p_hit": p_hit,
            "p_kill": p_kill,
            "events": events,
            "attention_maps": attention_maps or {},
            **kwargs
        }
        
        return self._generate_report_sync(request_data)