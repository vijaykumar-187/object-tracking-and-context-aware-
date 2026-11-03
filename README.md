# PERSON RE-ID | VISION ANALYTICS

Phase 2 prototype for daytime person detection and crop generation from up to three camera videos. The local Streamlit dashboard uses the existing `yolo26n.pt` model and OpenCV. Videos are processed one at a time with CPU inference.

## Phase 2 features

- Upload a video for Camera 1, Camera 2, and/or Camera 3 (each camera is optional).
- Sample frames at a configurable interval (30-frame default for quicker demos) and confidence threshold.
- Detect the COCO `person` class with the existing YOLO implementation.
- Save a processed MP4 with boxes held between checks; preview output is limited to 1280 pixels wide while crops use the original frames.
- Save clipped person crops, CSV detection metadata, and per-camera statistics.
- View progress, video results, crop galleries, and download camera or complete result ZIP files.

This prototype does **not** create persistent person IDs, Re-ID embeddings, cross-camera matches, tracking, or camera context.

## Requirements

- Windows 10 or 11
- Python 3.12 (the project was developed with Python 3.12.10)
- CPU-capable PyTorch, OpenCV, Ultralytics, and Streamlit, installed from `requirements.txt`
- YOLO weights at the project root: `yolo26n.pt`

The model file is ignored by Git because it is a binary weight file. Obtain the project’s supplied `yolo26n.pt` and place it beside `app.py` before launching. Model loading fails with a clear message if the file is absent.

## Install on Windows (PowerShell)

Open PowerShell in the project directory and run:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a virtual environment already exists, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Use an environment with CPU-capable PyTorch. The application explicitly sets YOLO inference to `device="cpu"`; it does not require CUDA.

## Run the dashboard

From the activated environment, in the project directory:

```powershell
python -m streamlit run app.py
```

Streamlit opens the local dashboard in your browser. In the dashboard, choose one or more camera videos, set the sampling interval and confidence threshold, then select **START PROCESSING**. You can preview the uploads before processing. Common MP4, AVI, MOV, MKV, and M4V files are accepted when the local OpenCV build supports their codec.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The existing integration tests load the YOLO weights and use CPU inference. New processor tests use short synthetic videos and mocked detections so they do not depend on model predictions.

## Results and logs

Each run is saved under:

```text
outputs/phase2_results/run_YYYYMMDD_HHMMSS/
```

Each processed camera has its own folder containing the input copy, `processed_video.mp4`, `detections.csv`, `statistics.json`, and `crops/` images. A complete ZIP is generated from the run folder when results are displayed. Technical errors are logged to `outputs/phase2_app.log`; normal dashboard errors are shown in plain language.

## How it works

- **Dashboard:** `app.py` builds the dark Streamlit UI, accepts optional camera uploads, shows status and results, and provides downloads.
- **Detection:** `src/detection/detect.py` remains the source for loading YOLO and detecting person boxes. The cached dashboard model is shared across all videos in a Streamlit process.
- **Video processing:** `src/detection/video_processor.py` validates each video, reads frames, invokes the detector at the selected interval, keeps the last observed boxes visible between checks, clips box coordinates, writes annotated video frames, saves non-empty crops, and reports measured counts and elapsed time.
- **Progress and statistics:** OpenCV's frame count is used for progress when available. Frame counters are shown if it is not. Output videos include every source frame; boxes and crops are created only for sampled frames, so the application does not imply detections on frames it did not analyze.

## Known limitations

- CPU-only inference can take time, especially for long or high-resolution videos. Boxes remain in their last detected position between sampled checks, so they can shift when YOLO refreshes them.
- OpenCV codec support depends on the local Windows installation. If a video cannot be opened or MP4 output cannot be written, the dashboard reports the issue.
- Streamlit holds uploaded files in memory before processing. Avoid very large files on laptops with limited memory.
- The sampling interval can miss people who appear only between sampled frames.
- The saved crop is a detection crop, not an identity or verified match.
