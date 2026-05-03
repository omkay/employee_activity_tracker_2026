# Employee Activity Tracking — research notebook

This project is an end-to-end computer-vision pipeline that watches office surveillance
video and produces structured activity events per employee (presence at desk, phone
usage, colleague interactions).

Two equivalent notebooks are provided — pick whichever workflow you prefer:

| File | Flavour | How to run |
|------|---------|------------|
| `employee_activity_tracking.ipynb` | Classic Jupyter — sequential, linear, great for research writeups | Open in Jupyter / VS Code and run top-to-bottom |
| `employee_activity_tracking_marimo.py` | **Reactive** marimo notebook — live sliders for thresholds, a dropdown to pick the video, a "▶ Run" button, and a filterable event explorer | `marimo edit employee_activity_tracking_marimo.py` (IDE) or `marimo run …` (app) |

## Pipeline at a glance

```
YOLOv8 (person + phone + laptop)
     │
     ▼
ByteTrack  ──► track_id (short-term)
     │
     ▼
InsightFace (face embedding)  +  OSNet / TorchReID (appearance embedding)
     │
     ▼
IdentityFuser (temporal voting over gallery matches) ──► employee_id (long-term)
     │
     ▼
Event engine (zones / phone / proximity state-machines) ──► events.csv
     │
     ▼
Annotated MP4  +  timeline plots  +  interaction graph
```

## Folder layout

```
graduation project/
├── employee_activity_tracking.ipynb   # the notebook
├── README.md                          # this file
├── data/                              # your input videos (.mp4)
├── gallery/                           # enrolled employees (see below)
│   ├── alice/
│   │   ├── face/  alice_01.jpg ...
│   │   └── body/  alice_body_01.jpg ...
│   └── bob/
│       ├── face/  ...
│       └── body/  ...
├── models/                            # cached model weights
└── outputs/                           # annotated videos + events csv/json
```

## Recommended Python & library versions

**Python 3.11** is the sweet spot — every dependency below has pre-built wheels for
it and the stack has been tested against it. Python 3.10 works too. Avoid 3.12+: the
`torchreid` wheel on PyPI is from 2020 and lacks 3.12 support; avoid 3.9 because
Ultralytics deprecation warnings start here.

| Package | Pin | Why this version |
|---------|-----|------------------|
| `python` | **3.11.x** | Broadest wheel coverage for this stack |
| `numpy` | `<2.0` | onnxruntime 1.18 & insightface 0.7 are not ABI-compatible with NumPy 2 |
| `pandas` | `==2.2.2` | Stable, matches plot code |
| `matplotlib` | `==3.9.0` | — |
| `seaborn` | `==0.13.2` | — |
| `opencv-python-headless` | `==4.10.0.84` | Avoid the non-headless variant on servers |
| `torch` | `==2.3.1` | Last version before 2.4's NumPy 2 requirement |
| `torchvision` | `==0.18.1` | Matched to torch 2.3.1 |
| `ultralytics` | `==8.2.50` | Known-good YOLOv8 + ByteTrack |
| `insightface` | `==0.7.3` | ArcFace / buffalo_l bundle |
| `onnxruntime` (CPU) / `onnxruntime-gpu` | `==1.18.0` | Pick GPU variant only if you have CUDA 12.x |
| `torchreid` | `==0.2.5` | OSNet implementation |
| `shapely` | `==2.0.4` | Zone geometry |
| `scikit-learn` | `==1.5.0` | Used by torchreid internals |
| `networkx` | `>=3.2` | Interaction graph |
| `tqdm` | `==4.66.4` | Progress bars |
| `marimo` | `>=0.23` | Reactive notebook |
| `jupyterlab` | `>=4.2` | For the `.ipynb` variant |

## Installation

Pick **one** of the three flavours below depending on your hardware.

### A. CPU-only (fastest to set up, slow at inference ~3 fps)

```bash
# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# core deep-learning stack (CPU)
pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu

# pipeline libs
pip install "numpy<2" "pandas==2.2.2" "matplotlib==3.9.0" "seaborn==0.13.2" \
            "opencv-python-headless==4.10.0.84" "tqdm==4.66.4" \
            "shapely==2.0.4" "scikit-learn==1.5.0" "networkx>=3.2"

# models
pip install "ultralytics==8.2.50" "insightface==0.7.3" "onnxruntime==1.18.0" \
            "torchreid==0.2.5"

# notebooks
pip install "marimo>=0.23" "jupyterlab>=4.2" "nbformat>=5.10"
```

### B. NVIDIA GPU (CUDA 12.1) — recommended for real runs

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121

pip install "numpy<2" "pandas==2.2.2" "matplotlib==3.9.0" "seaborn==0.13.2" \
            "opencv-python-headless==4.10.0.84" "tqdm==4.66.4" \
            "shapely==2.0.4" "scikit-learn==1.5.0" "networkx>=3.2"

pip install "ultralytics==8.2.50" "insightface==0.7.3" "onnxruntime-gpu==1.18.0" \
            "torchreid==0.2.5"

pip install "marimo>=0.23" "jupyterlab>=4.2" "nbformat>=5.10"
```

> Substitute `cu118` / `onnxruntime-gpu==1.18.0` if you are on CUDA 11.8.
> `nvidia-smi` must report a driver supporting your chosen CUDA version.

### C. Conda (nice if you already live in Anaconda/Miniforge)

```bash
conda create -n empact python=3.11 -y
conda activate empact

# PyTorch via conda (picks the right CUDA automatically; use cpuonly for CPU)
conda install pytorch=2.3.1 torchvision=0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# or:  conda install pytorch=2.3.1 torchvision=0.18.1 cpuonly -c pytorch -y

pip install "numpy<2" "pandas==2.2.2" "matplotlib==3.9.0" "seaborn==0.13.2" \
            "opencv-python-headless==4.10.0.84" "tqdm==4.66.4" \
            "shapely==2.0.4" "scikit-learn==1.5.0" "networkx>=3.2" \
            "ultralytics==8.2.50" "insightface==0.7.3" "onnxruntime-gpu==1.18.0" \
            "torchreid==0.2.5" "marimo>=0.23" "jupyterlab>=4.2" "nbformat>=5.10"
```

### D. Single-file shortcut (uses `requirements.txt` in this folder)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1) Install torch FIRST with the wheel matching your hardware
pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cpu
# or for NVIDIA CUDA 12.1:
# pip install "torch==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121

# 2) Install everything else
pip install -r requirements.txt
```

### E. One-shot with `uv` (experimental but fastest — ~30 s cold install)

The marimo file carries PEP-723 inline deps, so you can just:

```bash
pip install uv
uv run employee_activity_tracking_marimo.py        # auto-creates an env and runs
# or edit mode:
uv run marimo edit employee_activity_tracking_marimo.py
```

### Verify the install

```bash
python -c "
import sys, torch, numpy, cv2, ultralytics, insightface, torchreid, onnxruntime, marimo
print('python      :', sys.version.split()[0])
print('torch       :', torch.__version__, '| cuda:', torch.cuda.is_available())
print('numpy       :', numpy.__version__)
print('opencv      :', cv2.__version__)
print('ultralytics :', ultralytics.__version__)
print('insightface :', insightface.__version__)
print('onnxruntime :', onnxruntime.__version__)
print('torchreid   :', torchreid.__version__)
print('marimo      :', marimo.__version__)
"
```

Expect all imports to succeed and `torch.cuda.is_available()` to be `True` on a GPU box.

## Run the notebooks

```bash
# Jupyter flavour
jupyter lab employee_activity_tracking.ipynb

# Marimo flavour — interactive editor
marimo edit employee_activity_tracking_marimo.py

# Marimo flavour — read-only app view (good for demos)
marimo run employee_activity_tracking_marimo.py
```

## Workflow

1. **Enroll employees** — drop reference photos into `gallery/<name>/face/` and
   `gallery/<name>/body/`, then run the enrollment cell (Jupyter) or click the
   *Build / refresh gallery* button (marimo) to produce `gallery.npz`.
2. **Run the pipeline** — pick a video, set `max_frames` to something small for a
   smoke test, run. You'll get an annotated MP4 and an events CSV in `outputs/`.
3. **Inspect** — the Gantt timeline, per-employee totals, and interaction graph update
   reactively in the marimo version; re-run the plot cells in Jupyter.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ImportError: numpy.core.multiarray failed to import` | You have NumPy 2.x — `pip install "numpy<2"` |
| `onnxruntime` complaining about CUDA provider | You installed the CPU variant but asked for GPU — swap for `onnxruntime-gpu` |
| `torchreid` install errors | `pip install --no-deps torchreid` — it has an over-aggressive torch pin |
| `yolov8m.pt` download stalls | Pre-download from [ultralytics releases](https://github.com/ultralytics/assets/releases) and drop into `models/` |
| `insightface` model download fails | It auto-downloads to `~/.insightface`; pre-fetch the `buffalo_l` bundle |
| Ultra-slow on CPU | Set `FRAME_STRIDE=3–5` and lower `max_frames`; also consider `yolov8n.pt` |

## Stack
- PyTorch, Ultralytics YOLOv8, ByteTrack (built-in)
- InsightFace (buffalo_l / ArcFace)
- TorchReID (OSNet x1_0)
- shapely for zone geometry, matplotlib + seaborn for dashboards

## Notes & caveats
- All thresholds live in the **Configuration** cell (Section 1.1). Tune them for your
  camera geometry.
- The `DEFAULT_ZONES` in Section 6 are placeholders — redraw them over a frame from
  your camera before running on real footage.
- This is a research prototype. Deploying it in a real workplace requires written
  notice to employees, consent where legally required, short retention windows, and
  strict access control. Detection of "phone use" or "interaction" should inform
  decisions, not automate them.
