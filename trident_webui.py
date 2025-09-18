"""
TRIDENT-Net GUI (Gradio)

Dark-mode, modern UI to select train modes and directories, preview dataset,
and run training/evaluation simulations with live metrics. This is a Gradio
alternative to the Streamlit GUI with identical functionality.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
import tempfile
import uuid
import queue
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
import pandas as pd

# --- Theming ---
APPLE_DARK = {
    "bg": "#0b0b0d",
    "panel": "#151518",
    "muted": "#1e1e22",
    "text": "#eaeaea",
    "subtle": "#a8a8ad",
    "accent": "#5ac8fa",
    "good": "#30d158",
    "warn": "#ffd60a",
    "bad": "#ff453a",
}

APPLE_LIGHT = {
    "bg": "#f5f5f7",
    "panel": "#ffffff",
    "muted": "#f2f2f2",
    "text": "#1d1d1f",
    "subtle": "#6e6e73",
    "accent": "#007aff",
    "good": "#34c759",
    "warn": "#ff9f0a",
    "bad": "#ff3b30",
}

# Global state management
class AppState:
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.stop_flag = False
        self.drive_sa_temp: Optional[str] = None
        self.logs: List[str] = []
        self.chart_data: List[Dict[str, Any]] = []
        self.running = False

    def reset_process(self):
        self.proc = None
        self.stop_flag = False
        self.logs = []
        self.chart_data = []
        self.running = False

app_state = AppState()

def get_theme_css(dark: bool) -> str:
    """Generate CSS for dark/light theme."""
    P = APPLE_DARK if dark else APPLE_LIGHT
    return f"""
    <style>
    .gradio-container {{
        background: {P['bg']} !important;
        color: {P['text']} !important;
    }}
    .apple-title {{
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        line-height: 1.35;
        color: {P['text']};
        margin-bottom: 0.5rem;
    }}
    .apple-subtle {{
        color: {P['subtle']};
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .metric-pill {{
        background: rgba(127,127,127,0.12);
        border-radius: 999px;
        padding: 8px 12px;
        display: inline-block;
        margin: 4px 8px 4px 0;
        white-space: nowrap;
    }}
    .accent {{ color: {P['accent']}; }}
    .good {{ color: {P['good']}; }}
    .warn {{ color: {P['warn']}; }}
    .bad {{ color: {P['bad']}; }}
    </style>
    """

def read_prompts_jsonl(path: Path) -> pd.DataFrame:
    """Read prompts.jsonl file and return DataFrame."""
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception:
        pass
    if not rows:
        return pd.DataFrame()
    
    # Pull key fields for quick glance
    flat = []
    for r in rows:
        video = (r.get("video") or {})
        flat.append({
            "timestamp": r.get("timestamp_utc"),
            "scenario": r.get("scenario"),
            "bearing": (r.get("selections") or {}).get("bearing_deg"),
            "elevation": (r.get("selections") or {}).get("elevation_deg"),
            "range_km": (r.get("selections") or {}).get("range_km"),
            "rgb": video.get("path"),
            "ir": video.get("ir_path"),
            "fps": video.get("fps"),
        })
    return pd.DataFrame(flat)

def list_videos(root: Path) -> pd.DataFrame:
    """List video files in directory and check for IR availability."""
    rgb_files = sorted([p.name for p in root.glob("*.mp4")])
    ir_dir = root / "ir"
    
    # Build IR set supporting two conventions: same basename or *_ir.mp4
    ir_basenames = set()
    if ir_dir.exists():
        for p in ir_dir.glob("*.mp4"):
            name = p.name
            if name.endswith("_ir.mp4"):
                ir_basenames.add(name[:-7] + ".mp4")  # strip _ir
            else:
                ir_basenames.add(name)
    
    df = pd.DataFrame({"rgb": rgb_files})
    df["ir_available"] = df["rgb"].apply(lambda x: x in ir_basenames)
    return df

def preview_dataset(train_dir: str, eval_dir: str) -> Tuple[str, str, str, str]:
    """Preview train and eval datasets."""
    # Train dataset preview
    troot = Path(train_dir)
    t_prompts = troot / "prompts.jsonl"
    
    train_prompts_html = ""
    train_videos_html = ""
    
    if t_prompts.exists():
        try:
            df_t = read_prompts_jsonl(t_prompts)
            if not df_t.empty:
                train_prompts_html = df_t.to_html(classes="dataset-table", escape=False)
            else:
                train_prompts_html = "<p>prompts.jsonl is empty</p>"
        except Exception as e:
            train_prompts_html = f"<p>Error reading prompts.jsonl: {e}</p>"
    else:
        train_prompts_html = "<p>prompts.jsonl not found in Train/</p>"
    
    if troot.exists():
        try:
            files_t = list_videos(troot)
            if not files_t.empty:
                train_videos_html = files_t.to_html(classes="dataset-table", escape=False)
            else:
                train_videos_html = "<p>No video files found</p>"
        except Exception as e:
            train_videos_html = f"<p>Error listing videos: {e}</p>"
    else:
        train_videos_html = "<p>Train directory does not exist</p>"
    
    # Eval dataset preview
    eroot = Path(eval_dir)
    e_prompts = eroot / "prompts.jsonl"
    
    eval_prompts_html = ""
    eval_videos_html = ""
    
    if e_prompts.exists():
        try:
            df_e = read_prompts_jsonl(e_prompts)
            if not df_e.empty:
                eval_prompts_html = df_e.to_html(classes="dataset-table", escape=False)
            else:
                eval_prompts_html = "<p>prompts.jsonl is empty</p>"
        except Exception as e:
            eval_prompts_html = f"<p>Error reading prompts.jsonl: {e}</p>"
    else:
        eval_prompts_html = "<p>prompts.jsonl not found in Eval/</p>"
    
    if eroot.exists():
        try:
            files_e = list_videos(eroot)
            if not files_e.empty:
                eval_videos_html = files_e.to_html(classes="dataset-table", escape=False)
            else:
                eval_videos_html = "<p>No video files found</p>"
        except Exception as e:
            eval_videos_html = f"<p>Error listing videos: {e}</p>"
    else:
        eval_videos_html = "<p>Eval directory does not exist</p>"
    
    return train_prompts_html, train_videos_html, eval_prompts_html, eval_videos_html

def run_cli_and_stream(
    command: List[str],
    env_overrides: Dict[str, str] | None = None,
    env_unset: List[str] | None = None,
) -> str:
    """
    Run a CLI command and stream stdout lines.
    Parses simple 'loss', 'AUROC', 'F1' numbers if present and updates charts.
    """
    global app_state
    
    # Ensure no previous process is running
    if app_state.proc and app_state.proc.poll() is None:
        return "A process is already running. Stop it first."
    
    app_state.reset_process()
    app_state.running = True
    
    # Assemble environment to avoid buffering
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    
    # Remove specific env vars for this run if requested
    if env_unset:
        for k in env_unset:
            env.pop(str(k), None)
    
    # Apply per-run environment overrides
    if env_overrides:
        for k, v in env_overrides.items():
            env[str(k)] = str(v)
    
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
    except Exception as e:
        app_state.running = False
        return f"Failed to start process: {e}"
    
    app_state.proc = proc
    
    # Start log reader thread
    def log_reader():
        loss_re = re.compile(r"loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
        auroc_re = re.compile(r"AUROC\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
        f1_re = re.compile(r"F1\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
        
        try:
            for line in iter(proc.stdout.readline, ''):
                if not line:
                    break
                line = line.rstrip('\n')
                if line:
                    app_state.logs.append(line)
                    
                    # Parse metrics
                    m_loss = loss_re.search(line)
                    m_auroc = auroc_re.search(line)
                    m_f1 = f1_re.search(line)
                    
                    if m_loss or m_auroc or m_f1:
                        step = len(app_state.chart_data) + 1
                        rec: Dict[str, Any] = {"step": step}
                        if m_loss:
                            rec["loss"] = float(m_loss.group(1))
                        if m_auroc:
                            rec["AUROC"] = float(m_auroc.group(1))
                        if m_f1:
                            rec["F1"] = float(m_f1.group(1))
                        app_state.chart_data.append(rec)
        except Exception:
            pass
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
    
    reader_thread = threading.Thread(target=log_reader, daemon=True)
    reader_thread.start()
    
    return "Process started successfully"

def update_logs_and_charts():
    """Update logs and charts from running process."""
    global app_state
    
    if not app_state.proc:
        return "", None, "No process"
    
    # Check if process has finished
    if app_state.proc.poll() is not None and app_state.running:
        app_state.running = False
        rc = app_state.proc.returncode
        if rc and rc != 0:
            app_state.logs.append(f"Process exited with code {rc}")
        else:
            app_state.logs.append("Process completed successfully")
    
    # Prepare chart data
    chart_plot = None
    if app_state.chart_data:
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        
        chart_df = pd.DataFrame(app_state.chart_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'loss' in chart_df.columns:
            ax.plot(chart_df['step'], chart_df['loss'], label='Loss', color='#ff453a')
        if 'AUROC' in chart_df.columns:
            ax.plot(chart_df['step'], chart_df['AUROC'], label='AUROC', color='#30d158')
        if 'F1' in chart_df.columns:
            ax.plot(chart_df['step'], chart_df['F1'], label='F1', color='#5ac8fa')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title('Training Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        chart_plot = fig
        plt.close(fig)
    
    # Return recent logs and chart
    recent_logs = "\n".join(app_state.logs[-50:])  # Show last 50 lines
    status = "Running" if app_state.running else "Stopped"
    
    return recent_logs, chart_plot, status

def stop_process():
    """Stop the running process."""
    global app_state
    
    if app_state.proc and app_state.proc.poll() is None:
        try:
            app_state.proc.terminate()
            app_state.stop_flag = True
            app_state.running = False
            return "Process stopped"
        except Exception as e:
            return f"Error stopping process: {e}"
    return "No process running"

def start_training(
    mode: str, train_dir: str, eval_dir: str, config_path: str, pipeline: str,
    training_engine: str, use_synth: bool, batch_size: int, num_workers: int, pin_memory: bool,
    grad_accum_steps: int, ckpt_policy: str, ckpt_steps: int, drive_dir: str, use_drive_api: bool,
    drive_folder_id: str, device_choice: str
) -> str:
    """Start the training/evaluation process."""
    global app_state
    
    if app_state.running:
        return "A process is already running. Stop it first."
    
    # Build command
    py = sys.executable
    
    # Choose CLI module based on training engine
    if training_engine == "Memory-Efficient":
        cli_module = "trident.runtime.memory_efficient_cli"
    else:
        cli_module = "trident.runtime.cli"
    
    if mode == "train" or mode == "finaltrain":
        if training_engine == "Memory-Efficient":
            # Use memory-efficient CLI
            cmd = [
                py, "-m", cli_module,
                "--config", config_path,
                "--use-fp16",  # Enable FP16 by default
                "--checkpoint-every-layer",  # Enable checkpointing
                "--grad-accum-steps", str(int(grad_accum_steps)),  # Configurable gradient accumulation
                "--optimizer", "adamw8bit",  # Use 8-bit optimizer
                "--zero-stage", "2",  # DeepSpeed ZeRO-2 by default
            ]
        else:
            # Use standard CLI
            cmd = [
                py, "-m", cli_module, "train",
                "--config", config_path,
                "--pipeline", pipeline,
            ]
        
        if use_synth:
            cmd.append("--synthetic")
        
        # Loader overrides
        cmd += ["--batch-size", str(int(batch_size)), "--num-workers", str(int(num_workers))]
        cmd += (["--pin-memory"] if pin_memory else ["--no-pin-memory"])
        
        # Standard CLI specific options
        if training_engine == "Standard":
            # Checkpointing
            cmd += ["--ckpt-policy", ckpt_policy]
            if ckpt_policy == "steps" and int(ckpt_steps) > 0:
                cmd += ["--ckpt-steps", str(int(ckpt_steps))]
            
            # Google Drive mirror
            if drive_dir.strip():
                cmd += ["--drive-dir", drive_dir.strip()]
            
            # Google Drive API upload
            if use_drive_api and drive_folder_id.strip() and app_state.drive_sa_temp:
                cmd += ["--drive-api-folder-id", drive_folder_id.strip(),
                        "--drive-service-account", app_state.drive_sa_temp]
        
        # Dataset overrides
        t_prompts = str((Path(train_dir) / "prompts.jsonl").resolve())
        if Path(t_prompts).exists():
            cmd += ["--jsonl", t_prompts]
        cmd += ["--video-root", str(Path(train_dir).resolve())]
        
    elif mode == "eval":
        if training_engine == "Memory-Efficient":
            # Note: Evaluation with memory-efficient engine is not fully supported yet
            # Fall back to standard CLI for evaluation
            cli_module = "trident.runtime.cli"
            app_state.last_output = "⚠️ Note: Using Standard engine for evaluation (Memory-Efficient eval not yet supported)"
        
        cmd = [
            py, "-m", cli_module, "eval",
            "--config", config_path,
        ]
        if use_synth:
            cmd.append("--synthetic")
        
        # Loader overrides
        cmd += ["--batch-size", str(int(batch_size)), "--num-workers", str(int(num_workers))]
        cmd += (["--pin-memory"] if pin_memory else ["--no-pin-memory"])
        
        # The CLI automatically loads the best checkpoint from the latest training run
        # No need to specify --checkpoint parameter
        
        e_prompts = str((Path(eval_dir) / "prompts.jsonl").resolve())
        if Path(e_prompts).exists():
            cmd += ["--jsonl", e_prompts]
        cmd += ["--video-root", str(Path(eval_dir).resolve())]
    else:
        cmd = [py, "-m", cli_module, mode, "--config", config_path]
    
    # Per-run environment overrides
    overrides = {"CUDA_VISIBLE_DEVICES": "-1"} if device_choice == "CPU" else None
    unset = ["CUDA_VISIBLE_DEVICES"] if device_choice != "CPU" else None
    
    result = run_cli_and_stream(cmd, env_overrides=overrides, env_unset=unset)
    return result

def save_service_account_file(file) -> str:
    """Save uploaded service account file."""
    global app_state
    
    if file is None:
        return "No file uploaded"
    
    try:
        tmp_path = Path(tempfile.gettempdir()) / f"trident_sa_{uuid.uuid4().hex}.json"
        with open(tmp_path, 'wb') as f:
            f.write(file)
        app_state.drive_sa_temp = str(tmp_path)
        return f"Saved credentials to: {tmp_path}"
    except Exception as e:
        return f"Failed to save credentials: {e}"

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        title="TRIDENT‑Net Studio", 
        theme=gr.themes.Base(),
        css=get_theme_css(True),  # Default to dark theme
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="apple-title">🛰️ TRIDENT‑Net Studio</div>
        <div class="apple-subtle">Multimodal fusion training & evaluation. Select a mode, choose datasets, preview clips, and monitor metrics in real time.</div>
        """)
        
        # Main configuration row
        with gr.Row():
            with gr.Column(scale=2):
                mode = gr.Dropdown(
                    choices=["train", "eval", "finaltrain"],
                    value="train",
                    label="Mode"
                )
            with gr.Column(scale=2):
                train_dir = gr.Textbox(
                    value=str((Path.cwd() / "Train").resolve()),
                    label="Train directory"
                )
            with gr.Column(scale=2):
                eval_dir = gr.Textbox(
                    value=str((Path.cwd() / "Eval").resolve()),
                    label="Eval directory"
                )
            with gr.Column(scale=2):
                dark_mode = gr.Checkbox(
                    value=True,
                    label="Dark mode"
                )
        
        # Config settings
        with gr.Row():
            with gr.Column(scale=3):
                config_path = gr.Textbox(
                    value=str((Path.cwd() / "tasks.yml").resolve()),
                    label="Config (tasks.yml)"
                )
            with gr.Column(scale=2):
                pipeline = gr.Dropdown(
                    choices=["normal", "finaltrain"],
                    value="normal",
                    label="Pipeline"
                )
            with gr.Column(scale=2):
                training_engine = gr.Dropdown(
                    choices=["Standard", "Memory-Efficient"],
                    value="Standard",
                    label="Training Engine",
                    info="Choose between standard training pipeline or memory-efficient training with optimizations"
                )
            with gr.Column(scale=2):
                use_synth = gr.Checkbox(
                    value=False,
                    label="Use synthetic"
                )
        
        # Memory-efficient training info panel
        memory_info = gr.HTML(
            value="",
            visible=False
        )
        
        def update_memory_info(engine):
            if engine == "Memory-Efficient":
                return gr.update(
                    value="""
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #2196f3;">
                        <h4 style="margin: 0 0 10px 0; color: #1976d2;">🧠 Memory-Efficient Training Active</h4>
                        <p style="margin: 5px 0; color: #424242;">This mode enables several optimizations for GPU memory constraints:</p>
                        <ul style="margin: 5px 0; color: #424242;">
                            <li><strong>FP16 Mixed Precision:</strong> ~50% memory reduction</li>
                            <li><strong>Activation Checkpointing:</strong> Trade computation for memory</li>
                            <li><strong>8-bit Optimizers:</strong> AdamW8bit for reduced optimizer states</li>
                            <li><strong>DeepSpeed ZeRO-2:</strong> CPU optimizer offload</li>
                            <li><strong>Gradient Accumulation:</strong> Micro-batching (4 steps default)</li>
                        </ul>
                        <p style="margin: 5px 0; color: #424242;"><em>Ideal for training on single GPU with &lt;39GB VRAM (e.g., A100-40GB).</em></p>
                    </div>
                    """,
                    visible=True
                )
            else:
                return gr.update(value="", visible=False)
        
        training_engine.change(
            update_memory_info,
            inputs=[training_engine],
            outputs=[memory_info]
        )
        
        gr.HTML("<hr>")
        
        # Loader settings
        gr.HTML('<div class="apple-title">Loader settings</div>')
        with gr.Row():
            batch_size = gr.Number(value=2, minimum=1, label="Batch size", precision=0)
            num_workers = gr.Number(value=0, minimum=0, label="Workers", precision=0)
            pin_memory = gr.Checkbox(value=False, label="Pin memory")
        
        # Memory-efficient training settings  
        gr.HTML('<div class="apple-title">Memory-Efficient Settings</div>')
        with gr.Row():
            grad_accum_steps = gr.Number(
                value=4, minimum=1, maximum=16, step=1,
                label="Gradient accumulation steps", precision=0,
                info="Number of gradient accumulation steps (lower = less memory, more frequent updates)"
            )
        
        # Checkpointing
        gr.HTML('<div class="apple-title">Checkpointing</div>')
        with gr.Row():
            with gr.Column(scale=2):
                ckpt_policy = gr.Dropdown(
                    choices=["both", "last_epoch", "best", "steps", "off"],
                    value="both",
                    label="Policy"
                )
            with gr.Column(scale=1):
                ckpt_steps = gr.Number(
                    value=0, minimum=0, step=100,
                    label="Steps interval"
                )
        
        # Google Drive
        gr.HTML('<div class="apple-title">Google Drive</div>')
        with gr.Row():
            drive_dir = gr.Textbox(
                value="", 
                label="Local Drive sync folder (optional)"
            )
            use_drive_api = gr.Checkbox(
                value=False,
                label="Upload via Drive API"
            )
        
        with gr.Row():
            drive_folder_id = gr.Textbox(
                value="",
                label="Drive Folder ID"
            )
            service_account_file = gr.File(
                label="Service account JSON",
                file_types=[".json"]
            )
        
        service_account_status = gr.Textbox(
            value="", 
            label="Service Account Status",
            interactive=False
        )
        
        # Dataset previews
        gr.HTML('<div class="apple-title">Dataset preview — Train</div>')
        with gr.Row():
            train_prompts_preview = gr.HTML(value="")
            train_videos_preview = gr.HTML(value="")
        
        gr.HTML('<div class="apple-title">Dataset preview — Eval</div>')
        with gr.Row():
            eval_prompts_preview = gr.HTML(value="")
            eval_videos_preview = gr.HTML(value="")
        
        preview_btn = gr.Button("Preview Datasets", variant="secondary")
        
        gr.HTML("<hr>")
        
        # Controls and metrics
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="apple-title">Controls</div>')
                device_choice = gr.Radio(
                    choices=["CUDA (GPU)", "CPU"],
                    value="CUDA (GPU)",
                    label="Device"
                )
                
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop", variant="secondary")
                
                gr.HTML('<div class="apple-subtle">Start will call TRIDENT runtime CLI with the selected mode and stream logs live.</div>')
                
                gr.HTML('<div class="apple-title" style="margin-top:18px">Status</div>')
                process_status = gr.Textbox(
                    value="Stopped", 
                    label="Process Status",
                    interactive=False
                )
                
                start_result = gr.Textbox(
                    value="", 
                    label="Start Result",
                    interactive=False
                )
            
            with gr.Column(scale=3):
                gr.HTML('<div class="apple-title">Live metrics</div>')
                
                with gr.Tabs():
                    with gr.Tab("Loss & AUROC"):
                        metrics_plot = gr.Plot(label="Loss and AUROC over time")
                    with gr.Tab("Logs"):
                        logs_display = gr.Textbox(
                            value="",
                            label="Process Logs",
                            lines=20,
                            max_lines=20,
                            interactive=False
                        )
                    with gr.Tab("Confusion"):
                        gr.HTML("Will appear here when implemented in CLI outputs.")
                    with gr.Tab("Calibration"):
                        gr.HTML("Will appear here when implemented in CLI outputs.")
        
        # Event handlers
        service_account_file.change(
            save_service_account_file,
            inputs=[service_account_file],
            outputs=[service_account_status]
        )
        
        preview_btn.click(
            preview_dataset,
            inputs=[train_dir, eval_dir],
            outputs=[train_prompts_preview, train_videos_preview, eval_prompts_preview, eval_videos_preview]
        )
        
        start_btn.click(
            start_training,
            inputs=[
                mode, train_dir, eval_dir, config_path, pipeline,
                training_engine, use_synth, batch_size, num_workers, pin_memory,
                grad_accum_steps, ckpt_policy, ckpt_steps, drive_dir, use_drive_api,
                drive_folder_id, device_choice
            ],
            outputs=[start_result]
        )
        
        stop_btn.click(
            stop_process,
            outputs=[start_result]
        )
        
        # Manual refresh button for logs and charts
        refresh_btn = gr.Button("Refresh Logs & Charts", variant="secondary")
        
        def auto_refresh():
            return update_logs_and_charts()
        
        refresh_btn.click(
            auto_refresh,
            outputs=[logs_display, metrics_plot, process_status]
        )
    
    return demo

def main():
    """Main function to launch the Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )

if __name__ == "__main__":
    main()