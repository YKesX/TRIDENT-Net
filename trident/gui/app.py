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
import queue
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# --- Theming ---
st.set_page_config(page_title="TRIDENT-Net Studio", page_icon="🛰️", layout="wide")

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

def css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: linear-gradient(180deg, {APPLE_DARK['bg']} 0%, #0e0e11 100%); color:{APPLE_DARK['text']}; }}
        .block-container {{ padding-top: 1.5rem; }}
        .apple-card {{ background:{APPLE_DARK['panel']}; border-radius: 16px; padding: 16px 20px; border:1px solid {APPLE_DARK['muted']}; box-shadow: 0 8px 24px rgba(0,0,0,0.25); }}
        .apple-title {{ font-size: 1.25rem; font-weight:600; letter-spacing:0.2px; }}
        .apple-subtle {{ color:{APPLE_DARK['subtle']}; font-size:0.9rem; }}
        .metric-pill {{ background:{APPLE_DARK['muted']}; border-radius: 999px; padding: 8px 12px; display:inline-block; margin-right:8px; }}
        .accent {{ color:{APPLE_DARK['accent']}; }}
        .good {{ color:{APPLE_DARK['good']}; }}
        .warn {{ color:{APPLE_DARK['warn']}; }}
        .bad {{ color:{APPLE_DARK['bad']}; }}
    .file-grid {{ display:grid; grid-template-columns: repeat(4, minmax(200px,1fr)); gap:12px; }}
    .file-item {{ background:{APPLE_DARK['muted']}; border:1px solid {APPLE_DARK['panel']}; padding:10px; border-radius:12px; }}
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


def simulate_training(stream_placeholder, charts_placeholder, steps=50):
    import numpy as np
    logs = []
    loss = 1.0
    auroc = 0.5
    f1 = 0.3
    for i in range(steps):
        time.sleep(0.05)
        loss = max(0.02, loss * 0.97)
        auroc = min(0.99, auroc + np.random.rand() * 0.01 + 0.01)
        f1 = min(0.9, f1 + np.random.rand() * 0.015 + 0.005)
        logs.append({"step": i+1, "loss": loss, "AUROC": auroc, "F1": f1})
        stream_placeholder.write(f"Step {i+1}: loss {loss:.4f}, AUROC {auroc:.3f}, F1 {f1:.3f}")
        charts_placeholder.line_chart(pd.DataFrame(logs).set_index("step"))


def run_cli_and_stream(
    command: List[str],
    log_area,
    chart_area,
    stop_flag_key: str,
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
    css()

    st.markdown("<div class='apple-title'>TRIDENT‑Net Studio</div>", unsafe_allow_html=True)
    st.markdown("<div class='apple-subtle'>Select a mode, choose datasets, preview clips, and monitor metrics in real time.</div>", unsafe_allow_html=True)

    with st.container():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            mode = st.selectbox("Mode", ["train", "eval", "finaltrain"], index=0)
        with c2:
            train_dir = st.text_input("Train directory", value=str((Path.cwd() / "Train").resolve()))
        with c3:
            eval_dir = st.text_input("Eval directory", value=str((Path.cwd() / "Eval").resolve()))
        with c4:
            theme_ok = st.toggle("Dark mode", value=True, help="Dark mode is enabled by default")

    # Extra config: tasks.yml path and pipeline variant
    cfg_cols = st.columns([3, 2, 2])
    with cfg_cols[0]:
        config_path = st.text_input("Config (tasks.yml)", value=str((Path.cwd() / "tasks.yml").resolve()))
    with cfg_cols[1]:
        pipeline = st.selectbox("Pipeline", ["normal", "finaltrain"], index=0, help="Training sub-pipeline for CLI")
    with cfg_cols[2]:
        use_synth = st.toggle("Use synthetic", value=False)

    st.markdown("---")

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
                st.markdown("<div class='metric-pill'>Train: %s</div>" % train_dir, unsafe_allow_html=True)
                st.markdown("<div class='metric-pill'>Eval: %s</div>" % eval_dir, unsafe_allow_html=True)
            # Build command
            py = sys.executable
            if mode == "train" or mode == "finaltrain":
                cmd = [
                    py, "-m", "trident.runtime.cli", "train",
                    "--config", config_path,
                    "--pipeline", pipeline,
                ]
                if use_synth:
                    cmd.append("--synthetic")
                # Dataset overrides
                t_prompts = str((Path(train_dir) / "prompts.jsonl").resolve())
                if Path(t_prompts).exists():
                    cmd += ["--jsonl", t_prompts]
                cmd += ["--video-root", str(Path(train_dir).resolve())]
            elif mode == "eval":
                cmd = [
                    py, "-m", "trident.runtime.cli", "eval",
                    "--config", config_path,
                ]
                if use_synth:
                    cmd.append("--synthetic")
                e_prompts = str((Path(eval_dir) / "prompts.jsonl").resolve())
                if Path(e_prompts).exists():
                    cmd += ["--jsonl", e_prompts]
                cmd += ["--video-root", str(Path(eval_dir).resolve())]
            else:
                cmd = [py, "-m", "trident.runtime.cli", mode, "--config", config_path]

            run_cli_and_stream(cmd, stream, charts, stop_flag_key='stop_flag')

        if stop_btn and st.session_state.get('proc'):
            st.session_state.stop_flag = True

    with right:
        st.markdown("<div class='apple-title'>Live metrics</div>", unsafe_allow_html=True)
        tabs = st.tabs(["Loss & AUROC", "Confusion", "Calibration"])
        with tabs[0]:
            st.caption("Loss and AUROC over time")
            st.empty()  # Avoid empty chart warnings; charts update in left column
        with tabs[1]:
            st.caption("Confusion matrix (simulated)")
            st.image("https://dummyimage.com/600x240/151518/ffffff&text=Confusion+Matrix", width='stretch')
        with tabs[2]:
            st.caption("Reliability diagram (simulated)")
            st.image("https://dummyimage.com/600x240/151518/ffffff&text=Reliability+Diagram", width='stretch')


if __name__ == "__main__":
    main()
