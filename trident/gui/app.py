"""
TRIDENT-Net GUI (Streamlit)

Dark-mode, modern UI to select train modes and directories, preview dataset,
and run training/evaluation simulations with live metrics.
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
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# --- Theming ---
st.set_page_config(page_title="TRIDENT‑Net Studio", page_icon="🛰️", layout="wide")

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

def css(dark: bool):
    P = APPLE_DARK if dark else APPLE_LIGHT
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(180deg, {P['bg']} 0%, {P['bg']} 100%); color:{P['text']}; }}
        /* Push content below the top bar/header for visibility */
        .block-container {{ padding-top: 5.5rem !important; }}
        .apple-card {{ background:{P['panel']}; border-radius: 16px; padding: 16px 20px; border:1px solid rgba(127,127,127,0.18); box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
        .apple-title {{ font-size: 1.2rem; font-weight:600; letter-spacing:0.2px; line-height:1.35; white-space:normal; word-break:break-word; }}
        .apple-subtle {{ color:{P['subtle']}; font-size:0.95rem; line-height:1.5; white-space:normal; word-break:break-word; }}
        .metric-pill {{ background: rgba(127,127,127,0.12); border-radius: 999px; padding: 8px 12px; display:inline-block; margin: 4px 8px 4px 0; white-space:nowrap; }}
        .accent {{ color:{P['accent']}; }}
        .good {{ color:{P['good']}; }}
        .warn {{ color:{P['warn']}; }}
        .bad {{ color:{P['bad']}; }}
        .file-grid {{ display:grid; grid-template-columns: repeat(4, minmax(200px,1fr)); gap:12px; }}
        .file-item {{ background:rgba(127,127,127,0.08); border:1px solid rgba(127,127,127,0.18); padding:10px; border-radius:12px; }}
        /* Make dataframe cells wrap and not clip */
        .stDataFrame div {{ white-space:normal !important; word-break:break-word !important; }}
        .stMarkdown p {{ margin-bottom: 0.35rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_prompts_jsonl(path: Path) -> pd.DataFrame:
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


# simulate_training removed (unused)


def run_cli_and_stream(
    command: List[str],
    log_area,
    chart_area,
    stop_flag_key: str,
    env_overrides: Dict[str, str] | None = None,
    env_unset: List[str] | None = None,
):
    """
    Run a CLI command and stream stdout lines into Streamlit.
    Parses simple 'loss', 'AUROC', 'F1' numbers if present and updates a chart.
    """
    # Ensure no previous process is running
    if 'proc' in st.session_state and st.session_state.proc and st.session_state.proc.poll() is None:
        log_area.write("A process is already running. Stop it first.")
        return

    # Assemble environment to avoid buffering
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # Remove specific env vars for this run if requested (e.g., clear CUDA mask)
    if env_unset:
        for k in env_unset:
            env.pop(str(k), None)
    # Apply per-run environment overrides (e.g., force CPU)
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
        log_area.error(f"Failed to start process: {e}")
        return

    st.session_state.proc = proc
    st.session_state[stop_flag_key] = False

    # Regex parsers (used in main thread)
    loss_re = re.compile(r"loss\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    auroc_re = re.compile(r"AUROC\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)
    f1_re = re.compile(r"F1\s*[:=]\s*([0-9]*\.?[0-9]+)", re.I)

    # Thread-safe line queue
    q: "queue.Queue[str]" = queue.Queue(maxsize=1000)

    def reader():
        for line in iter(proc.stdout.readline, ''):
            try:
                q.put(line, timeout=0.5)
            except Exception:
                pass
        try:
            proc.stdout.close()
        except Exception:
            pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # UI state in main thread
    logs: List[Dict[str, Any]] = []
    chart_df = pd.DataFrame(columns=["loss", "AUROC", "F1"]).set_index(pd.Index([], name='step'))
    step = 0

    # Poll queue and update UI until process exits and queue drains
    while True:
        drained = 0
        try:
            while True:
                line = q.get_nowait()
                drained += 1
                line = line.rstrip('\n')
                if line:
                    log_area.write(line)
                    # Parse metrics
                    m_loss = loss_re.search(line)
                    m_auroc = auroc_re.search(line)
                    m_f1 = f1_re.search(line)
                    if m_loss or m_auroc or m_f1:
                        step += 1
                        rec: Dict[str, Any] = {"step": step}
                        if m_loss:
                            rec["loss"] = float(m_loss.group(1))
                        if m_auroc:
                            rec["AUROC"] = float(m_auroc.group(1))
                        if m_f1:
                            rec["F1"] = float(m_f1.group(1))
                        logs.append(rec)
                        try:
                            chart_df = pd.DataFrame(logs).set_index("step")
                            chart_area.line_chart(chart_df)
                        except Exception:
                            pass
        except queue.Empty:
            pass

        # Stop/terminate handling
        if st.session_state.get(stop_flag_key) and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        # Exit condition: process finished and queue drained
        if proc.poll() is not None:
            # drain remaining quickly
            try:
                while True:
                    line = q.get_nowait()
                    line = line.rstrip('\n')
                    if line:
                        log_area.write(line)
            except queue.Empty:
                break

        time.sleep(0.05 if drained == 0 else 0.0)

    # Ensure thread ends
    t.join(timeout=1.0)

    rc = proc.returncode
    if rc and rc != 0:
        log_area.error(f"Process exited with code {rc}")
    else:
        log_area.success("Done")


def main():
    st.markdown("<div class='apple-title'>🛰️ TRIDENT‑Net Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='apple-subtle'>Multimodal fusion training & evaluation. Select a mode, choose datasets, preview clips, and monitor metrics in real time.</div>", unsafe_allow_html=True)

    with st.container():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            mode = st.selectbox("Mode", ["train", "eval", "finaltrain"], index=0)
        with c2:
            train_dir = st.text_input("Train directory", value=str((Path.cwd() / "Train").resolve()))
        with c3:
            eval_dir = st.text_input("Eval directory", value=str((Path.cwd() / "Eval").resolve()))
    with c4:
        dark_mode = st.toggle("Dark mode", value=True, help="Toggle TRIDENT‑Net Studio dark/light theme")

    # Inject theme CSS after reading toggle
    css(dark_mode)

    # Extra config: tasks.yml path and pipeline variant
    cfg_cols = st.columns([3, 2, 2, 2])
    with cfg_cols[0]:
        config_path = st.text_input("Config (tasks.yml)", value=str((Path.cwd() / "tasks.yml").resolve()))
    with cfg_cols[1]:
        pipeline = st.selectbox("Pipeline", ["normal", "finaltrain"], index=0, help="Training sub-pipeline for CLI")
    with cfg_cols[2]:
        training_engine = st.selectbox(
            "Training Engine", 
            ["Standard", "Memory-Efficient"], 
            index=0, 
            help="Choose between standard training pipeline or memory-efficient training with optimizations (FP16, checkpointing, 8-bit optimizers)"
        )
    with cfg_cols[3]:
        use_synth = st.toggle("Use synthetic", value=False, help="Feed synthetic data batches into the CLI instead of reading dataset folders. Metrics are computed on these synthetic batches (no fake charts).")

    # Memory-efficient training info
    if training_engine == "Memory-Efficient":
        st.info("""
        🧠 **Memory-Efficient Training Active**
        
        This mode supports both GPU training and CPU evaluation systems:
        • **FP16 Mixed Precision**: ~50% memory reduction (auto-disabled on CPU)
        • **Activation Checkpointing**: Trade computation for memory
        • **8-bit Optimizers**: AdamW8bit for reduced optimizer states (auto-disabled on CPU)
        • **DeepSpeed ZeRO-2**: CPU optimizer offload (auto-disabled on CPU)
        • **Gradient Accumulation**: Micro-batching (4 steps default)
        • **CPU Compatibility**: Automatic fallback for CPU-only systems
        
        Training: A100 39GB + 70GB RAM | Evaluation: CPU-only + 30GB RAM
        """)

    st.markdown("---")

    # Loader overrides
    with st.container():
        st.markdown("<div class='apple-title'>Loader settings</div>", unsafe_allow_html=True)
        lc1, lc2, lc3 = st.columns([1,1,1])
        with lc1:
            bs_override = st.number_input("Batch size", min_value=1, value=2, step=1)
        with lc2:
            nw_override = st.number_input("Workers", min_value=0, value=0, step=1)
        with lc3:
            pinmem = st.checkbox("Pin memory", value=False)

    # Memory-efficient training settings
    with st.container():
        st.markdown("<div class='apple-title'>Memory-Efficient Settings</div>", unsafe_allow_html=True)
        mc1, mc2 = st.columns([1,1])
        with mc1:
            grad_accum_steps = st.number_input(
                "Gradient accumulation steps", 
                min_value=1, max_value=16, value=4, step=1,
                help="Number of gradient accumulation steps (lower = less memory, more frequent updates)"
            )
        with mc2:
            deepspeed_config = st.selectbox(
                "DeepSpeed Configuration",
                ["Auto (based on device)", 
                 "configs/a100_39gb_70gb_ram_training.json",
                 "configs/cpu_only_30gb_ram.json", 
                 "configs/a100_39gb_30gb_cpu.json"],
                index=0,
                help="Select hardware-specific configuration or let system auto-detect"
            )

    # Checkpointing
    with st.container():
        st.markdown("<div class='apple-title'>Checkpointing</div>", unsafe_allow_html=True)
        ck1, ck2 = st.columns([2,1])
        with ck1:
            ckpt_policy = st.selectbox("Policy", ["both", "last_epoch", "best", "steps", "off"], index=0,
                                       help="Choose when to save checkpoints: best metric, each epoch, both, fixed steps, or off.")
        with ck2:
            ckpt_steps = st.number_input("Steps interval", min_value=0, value=0, step=100,
                                        help="Used only when policy=steps. Save every N steps.")

    # Google Drive (local sync + API upload)
    with st.container():
        st.markdown("<div class='apple-title'>Google Drive</div>", unsafe_allow_html=True)
        gd1, gd2 = st.columns([2, 2])
        with gd1:
            drive_dir = st.text_input("Local Drive sync folder (optional)", value="",
                                      help="Path to your local Google Drive sync folder; checkpoints will be mirrored there.")
        with gd2:
            use_drive_api = st.checkbox("Upload via Drive API (service account)", value=False,
                                        help="Enable direct upload to a Drive folder using a service account.")
        api1, api2 = st.columns([2, 2])
        drive_folder_id = ""
        service_account_path = None
        if use_drive_api:
            with api1:
                drive_folder_id = st.text_input("Drive Folder ID", value="",
                                                help="Target folder ID in Google Drive (service account needs access).")
            with api2:
                sa_file = st.file_uploader("Service account JSON", type=["json"], accept_multiple_files=False)
                if sa_file is not None:
                    # Save uploaded creds to a temp file for this session
                    if 'drive_sa_temp' not in st.session_state:
                        st.session_state.drive_sa_temp = None
                    try:
                        tmp_path = Path(tempfile.gettempdir()) / f"trident_sa_{uuid.uuid4().hex}.json"
                        with open(tmp_path, 'wb') as f:
                            f.write(sa_file.read())
                        st.session_state.drive_sa_temp = str(tmp_path)
                        st.caption(f"Saved credentials to: {tmp_path}")
                    except Exception as e:
                        st.error(f"Failed to save credentials: {e}")
                service_account_path = st.session_state.get('drive_sa_temp')

    # Dataset preview (Train & Eval)
    with st.container():
        st.markdown("<div class='apple-title'>Dataset preview — Train</div>", unsafe_allow_html=True)
        troot = Path(train_dir)
        tcols = st.columns([2, 3])
        with tcols[0]:
            t_prompts = troot / "prompts.jsonl"
            if t_prompts.exists():
                df_t = read_prompts_jsonl(t_prompts)
                st.dataframe(df_t, width='stretch', height=260)
            else:
                st.info("prompts.jsonl not found in Train/.")
        with tcols[1]:
            if troot.exists():
                files_t = list_videos(troot)
                st.dataframe(files_t, width='stretch', height=260)
            else:
                st.warning("Train directory does not exist.")

    with st.container():
        st.markdown("<div class='apple-title'>Dataset preview — Eval</div>", unsafe_allow_html=True)
        eroot = Path(eval_dir)
        ecols = st.columns([2, 3])
        with ecols[0]:
            e_prompts = eroot / "prompts.jsonl"
            if e_prompts.exists():
                df_e = read_prompts_jsonl(e_prompts)
                st.dataframe(df_e, width='stretch', height=260)
            else:
                st.info("prompts.jsonl not found in Eval/.")
        with ecols[1]:
            if eroot.exists():
                files_e = list_videos(eroot)
                st.dataframe(files_e, width='stretch', height=260)
            else:
                st.warning("Eval directory does not exist.")

    st.markdown("---")

    # Actions & live metrics
    left, right = st.columns([2, 3])
    with left:
        st.markdown("<div class='apple-title'>Controls</div>", unsafe_allow_html=True)
        device_choice = st.radio(
            "Device",
            ["CUDA (GPU)", "CPU"],
            index=0,
            horizontal=True,
            help="Select compute device. CPU will run slower but sets CUDA_VISIBLE_DEVICES=-1 only for this run.",
        )
        run_btn = st.button("Start", type="primary")
        stop_btn = st.button("Stop", type="secondary")
        st.markdown("<div class='apple-subtle'>Start will call TRIDENT runtime CLI with the selected mode and stream logs live.</div>", unsafe_allow_html=True)

        st.markdown("<div class='apple-title' style='margin-top:18px'>Status</div>", unsafe_allow_html=True)
        stream = st.container()
        charts = st.empty()

        if 'proc' not in st.session_state:
            st.session_state.proc = None
        if 'stop_flag' not in st.session_state:
            st.session_state.stop_flag = False

        if run_btn and not st.session_state.get('proc'):
            with stream:
                st.markdown("<div class='metric-pill'>Pipeline: <span class='accent'>%s</span></div>" % mode, unsafe_allow_html=True)
                st.markdown("<div class='metric-pill'>Engine: <span class='accent'>%s</span></div>" % training_engine, unsafe_allow_html=True)
                st.markdown("<div class='metric-pill'>Train: %s</div>" % train_dir, unsafe_allow_html=True)
                st.markdown("<div class='metric-pill'>Eval: %s</div>" % eval_dir, unsafe_allow_html=True)
                st.markdown("<div class='metric-pill'>Device: %s</div>" % ("CPU" if device_choice == "CPU" else "CUDA (GPU)"), unsafe_allow_html=True)
            # Build command
            py = sys.executable
            
            # Choose CLI module based on training engine
            if training_engine == "Memory-Efficient":
                cli_module = "trident.runtime.memory_efficient_cli"
            else:
                cli_module = "trident.runtime.cli"
            
            if mode == "train" or mode == "finaltrain":
                if training_engine == "Memory-Efficient":
                    # Determine DeepSpeed config based on selection
                    if deepspeed_config == "Auto (based on device)":
                        # Auto-select based on device choice
                        if device_choice == "CPU":
                            selected_config = "configs/cpu_only_30gb_ram.json"
                        else:
                            selected_config = "configs/a100_39gb_70gb_ram_training.json"
                    else:
                        selected_config = deepspeed_config
                    
                    # Use memory-efficient CLI
                    cmd = [
                        py, "-m", cli_module,
                        "--config", config_path,
                        "--use-fp16",  # Enable FP16 by default
                        "--checkpoint-every-layer",  # Enable checkpointing
                        "--grad-accum-steps", str(int(grad_accum_steps)),  # Configurable gradient accumulation
                        "--optimizer", "adamw8bit",  # Use 8-bit optimizer
                        "--zero-stage", "2",  # DeepSpeed ZeRO-2 by default
                        "--deepspeed-config", selected_config,  # Use selected config
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
                cmd += ["--batch-size", str(int(bs_override)), "--num-workers", str(int(nw_override))]
                cmd += (["--pin-memory"] if pinmem else ["--no-pin-memory"])
                
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
                    if use_drive_api and drive_folder_id.strip() and service_account_path:
                        cmd += ["--drive-api-folder-id", drive_folder_id.strip(),
                                "--drive-service-account", service_account_path]
                
                # Dataset overrides
                t_prompts = str((Path(train_dir) / "prompts.jsonl").resolve())
                if Path(t_prompts).exists():
                    cmd += ["--jsonl", t_prompts]
                cmd += ["--video-root", str(Path(train_dir).resolve())]
            elif mode == "eval":
                if training_engine == "Memory-Efficient":
                    st.warning("⚠️ Evaluation mode not yet supported with Memory-Efficient engine. Using Standard engine for eval.")
                    cli_module = "trident.runtime.cli"
                
                cmd = [
                    py, "-m", cli_module, "eval",
                    "--config", config_path,
                ]
                if use_synth:
                    cmd.append("--synthetic")
                # Loader overrides
                cmd += ["--batch-size", str(int(bs_override)), "--num-workers", str(int(nw_override))]
                cmd += (["--pin-memory"] if pinmem else ["--no-pin-memory"])
                e_prompts = str((Path(eval_dir) / "prompts.jsonl").resolve())
                if Path(e_prompts).exists():
                    cmd += ["--jsonl", e_prompts]
                cmd += ["--video-root", str(Path(eval_dir).resolve())]
            else:
                cmd = [py, "-m", "trident.runtime.cli", mode, "--config", config_path]

            # Per-run environment overrides
            overrides = {"CUDA_VISIBLE_DEVICES": "-1"} if device_choice == "CPU" else None
            unset = ["CUDA_VISIBLE_DEVICES"] if device_choice != "CPU" else None
            run_cli_and_stream(cmd, stream, charts, stop_flag_key='stop_flag', env_overrides=overrides, env_unset=unset)

        if stop_btn and st.session_state.get('proc'):
            st.session_state.stop_flag = True

    with right:
        st.markdown("<div class='apple-title'>Live metrics</div>", unsafe_allow_html=True)
        tabs = st.tabs(["Loss & AUROC", "Confusion", "Calibration"])
        with tabs[0]:
            st.caption("Loss and AUROC over time")
            st.empty()  # Avoid empty chart warnings; charts update in left column
        with tabs[1]:
            st.caption("Confusion matrix")
            st.info("Will appear here when implemented in CLI outputs.")
        with tabs[2]:
            st.caption("Calibration")
            st.info("Will appear here when implemented in CLI outputs.")


if __name__ == "__main__":
    main()
