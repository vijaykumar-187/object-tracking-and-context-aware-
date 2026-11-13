# Person Re-ID | Review 3

A Windows and CPU-friendly multi-camera person detection, tracking, and appearance-matching application. The app reuses the existing YOLO26n detector. A separate OSNet x0.25 person-ReID model is available for appearance fingerprints; the YOLO feature baseline remains available for comparison.

## What the application does

- Add or remove up to 12 named camera sources and process each uploaded recorded video independently.
- Detect the COCO `person` class with the existing YOLO26n implementation on CPU.
- Track multiple people independently within each video using a Kalman motion model and Hungarian detection assignment. Track IDs are local to a camera video.
- Show camera, frame, local track ID, and observed or predicted box state on output videos.
- Inspect a chosen processed frame, select a track, and start, pause, resume, or stop its selected-person highlight. Inspect its recent history and see ACTIVE, LOST, or COMPLETED state.
- Filter small or blurry person crops and keep up to eight good, time-spread crops per track, including box coordinates and quality metadata.
- Save rejection reasons and a bounded set of small rejected-crop previews for visual inspection.
- Compare YOLO detector features with a separate OSNet x0.25 model trained for person Re-ID. OSNet embeddings use the published 256x128 RGB and ImageNet normalization preprocessing and run on CPU.
- Compare normalized mean, quality-weighted mean, and medoid track fingerprints. All three are built from one crop-embedding pass and each representative crop vector is saved with its frame and quality metadata. The existing mean remains the default until genuine labels show another method is better.
- Select any camera-local track and rank the top three candidates separately in every other processed camera, with an adjustable no-match threshold. A similarity score is only a ranking; it is not a probability or proof of identity.
- Save human-reviewed pair hypotheses as SAME PERSON, DIFFERENT PEOPLE, UNCERTAIN, or UNVERIFIED without changing camera-local IDs. Confirmed links are grouped into review-derived global IDs and flagged if the review decisions conflict.
- Evaluate with manually verified identity pairs, reporting top-1, top-3, top-5, mAP, MRR, evaluated identities, and missing fingerprints. mAP needs every same-person destination-camera tracklet in the label file.
- Download videos, crop images, CSV metadata, JSON summaries, and a ZIP of the run.

## Install the base application

Windows 10/11, Python 3.12, and the supplied `yolo26n.pt` file in the project root are required. In PowerShell, from the project folder:

```powershell
py -3.12 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, use the environment's Python executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

The project weight `yolo26n.pt` is ignored by Git because it is a binary file; copy the supplied detector weight beside `app.py`.

## Enable the dedicated OSNet Re-ID model

OSNet x0.25 is a lightweight model trained for person re-identification using MSMT17. It is not trained on your own camera videos. Torchreid's OSNet implementation projects the x0.25 backbone to a 512-value output feature. The downloaded checkpoint is not committed to Git.

With Git installed and an internet connection, in the same active environment:

```powershell
python -m pip install -r requirements-reid.txt
python tools/setup_reid_source.py
python tools/download_osnet_weights.py
```

The source setup fetches the official Python code without building Torchreid's optional Cython extension, avoiding a C++ compiler requirement on Windows. The weights are saved under `models/`. If the optional code, dependencies, or checkpoint is missing, the app says so and keeps the existing YOLO baseline. It does not pretend random or ImageNet-only weights are a trained Re-ID model. If OSNet is unavailable, top-three candidates and evaluation remain available for the YOLO baseline.

The code comes from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid); the pretrained checkpoint is in the [official OSNet model repository](https://huggingface.co/kaiyangzhou/osnet).

To verify the actual model with two saved crop images:

```powershell
python tools/verify_osnet.py path\to\person_crop_1.jpg path\to\person_crop_2.jpg
```

This reports model identity, CPU inference time, embedding shape, finite-value validation, and a cosine score. A meaningful score still needs known same-person/different-person crops for interpretation.

## Run the dashboard

```powershell
python -m streamlit run app.py
```

Add as many as 12 named camera inputs, choose videos, set detector interval, confidence, image size, NMS overlap, lost-track retention, association range, and crop quality settings, then select **Process camera videos**. Each camera is processed independently and keeps a separate local ID namespace. The dashboard has Overview, Camera workspace, Track management, Appearance matching, Evaluation, and Downloads tabs. In Appearance matching, select any source camera and local track ID; results are ranked separately for every other camera, and cameras without candidates show an explicit no-match state. You can seek to a processed frame and view the original uploaded filename. A smaller YOLO interval or larger image size usually improves detail but increases CPU time. The UI reports measured work and saves processing/performance summaries; it does not guarantee a fixed runtime.

The ground-truth CSV template for multi-camera evaluation is `query_camera_id,query_track_id,candidate_camera_id,candidate_track_id`. Add one same-person pair per row; candidate rankings are evaluated only against the specified destination camera. For a Camera 1 person absent from Camera 2, use `NOT_IN_CAMERA2` in `candidate_track_id`. The dashboard reports rank metrics, mAP, precision, recall, false/missed matches, and no-match rejection counts when suitable labels are supplied. Human-confirmed pairs are downloadable in this same format, but they only contain positive labels and do not by themselves measure no-match accuracy or provide a complete mAP gallery. Older two-camera pair CSVs are still accepted by the validation helper, but the dashboard exports the dynamic format. Global IDs are formed only from manually confirmed SAME PERSON links; they are run-local group labels, not model predictions.

### Create continuous selected-person videos

1. In **Appearance matching**, choose the source camera and its local track ID.
2. Review a candidate's crop and score for each destination camera. Select **SAME PERSON** and save to confirm that camera-local track. Similarity alone never creates a cross-camera association.
3. Select **Track This Person Across All Cameras**. The app assigns or reuses a persistent `GLOBAL-###` ID for the selected track, then writes one full-length MP4 for every processed camera.
4. Play or download each camera result. Confirmed tracks show only that person's box and the shared global ID. Short tracker predictions use a different color and are capped by the configured lost-track window. When no confirmed match exists, the original scene continues with a no-match label and no person box.

Generated files live in `outputs/review3_results/<run>/global_person_videos/<GLOBAL-ID>/`. Each folder contains the camera-specific playable videos, `video_manifest.json`, and `global_person_assignments.csv`. The registry and association evidence are included in the complete run ZIP. Human rejection removes that camera track from the selected global identity and invalidates its previous generated videos until they are generated again.

The selected-person **Start Tracking** and **Stop Tracking** controls highlight the chosen camera-local track in the processed video. The status distinguishes not tracking, tracking, temporarily lost, lost, stopped, and paused. The highlight stays bound to the selected run, camera, and track ID; it never changes to another person. Short predictions can be shown between detector checks, then the video labels the person as not detected when the prediction window ends. Processing is an offline batch operation; these controls render the selected target after the video has been processed.

## Repeatable two-camera Review 3 run

The standalone runner reuses YOLO26n, the existing video processor, and the cached OSNet feature adapter. It writes per-camera `fingerprints.npz`, JSON crop embeddings, a two-camera `matches.csv`, a `run_summary.json`, and the existing video/tracking exports. It reuses completed video processing only when the input paths, file size/timestamps, and processing settings still match.

```powershell
python -m src.reid.run_matching --camera1 data\videos\cam1.mp4 --camera2 data\videos\cam2.mp4 --output outputs\review3\latest
python -m streamlit run app_matching.py
```

The match viewer reads the finished run folder and shows the Camera 1 query crop with up to three Camera 2 candidates and scores. The current matching and evaluation default is the original normalized mean; quality-weighted and medoid variants are available for side-by-side labelled evaluation. No similarity cutoff is calibrated until real positive and negative pairs are provided.

### Measured checks in this checkout

- Two recorded clips (296 and 303 frames) were processed at 640 px, confidence 0.25, and a detector interval of 10 frames. Testing the tracker overlap gate at 0.05 and 0.30 yielded the same track counts: 16 for Camera 1 and 17 for Camera 2. Single-observation tracks were 2 and 0, respectively. The sweep did not demonstrate an accuracy improvement, so the roadmap default 0.05 is retained.
- Processing took 39.72 s and 29.67 s at 0.05, and 30.54 s and 30.50 s at 0.30. These are individual CPU runs on the available clips; runtime varied and is not guaranteed.
- The actual OSNet x0.25 checkpoint loaded on CPU in 27.21 s. A batch of 10 saved person crops produced finite 512D embeddings in 0.429 s (about 42.9 ms/crop); the Python process used about 378 MiB RSS after model loading.
- No complete ground-truth labels accompany these two clips, so ID switches, IDF1/HOTA, real Rank-1/5, mAP, false-match rate, and matching-threshold accuracy remain unmeasured. Synthetic tests validate program behavior only.

### Tracker and Re-ID model investigation

- The selected local tracker remains the small Kalman/Hungarian SORT-style implementation. The installed Ultralytics tracker package could not be imported in this offline environment because its `lap` dependency is missing. ByteTrack, BoT-SORT, and Deep OC-SORT therefore were not benchmarked; no newer tracker is claimed to be better.
- OSNet x0.25 stays the CPU Re-ID model. TransReID's official instructions use a much larger transformer setup and GPU-centric research dependencies; CLIP-ReID requires a two-stage Re-ID training procedure and task checkpoints. Neither was measured on these clips, so neither was added.
- NightReID provides relevant real nighttime RGB data and the EDA method, but its official release requires submitting an agreement before download. VAST-ReID's WACV 2026 paper describes low-light surveillance videos and identity-aligned crops, but the official paper page did not provide an accessible dataset download during this review. No results from either dataset are claimed.

## Dataset preparation: MEVID / MEVA

The official [Kitware MEVID release](https://github.com/Kitware/MEVID) links public S3 downloads for identity-labelled tracklet crops/metadata and 976 supporting five-minute recorded video clips (about 127 GB). MEVID labels 158 people across 33 camera views and 17 locations; its metadata includes global person ID, outfit ID, camera ID, and tracklet ID. The annotation package is separate from the original video clips. The [test crop/tracklet package (13 GB)](https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/mevid-v1-bbox-test.tgz), [annotation data](https://mevadata-public-01.s3.amazonaws.com/mevid-annotations/mevid-v1-annotation-data.zip), and [official video URL list](https://github.com/Kitware/MEVID/blob/main/mevid-v1-video-URLS.txt) are linked by the project. The [MEVA dataset site](https://mevadata.org/) describes its public no-cost S3 access and CC BY 4.0 terms.

After obtaining and unpacking one official `bbox_train` or `bbox_test` package and its annotation zip, create a two-camera ground-truth inventory and pair table:

```powershell
python tools/prepare_mevid_two_camera.py --root data\datasets\mevid --split test --camera1 329 --camera2 336
```

Add `--include-video-urls` to fetch the official list and write a separate CSV of public clips whose filenames identify either selected camera. The utility writes links only; it does not automatically download the large video corpus. Use camera IDs available in that split; the utility reports available IDs when a camera has no tracklets. It writes a tracklet inventory, a separate same-person/different-person pair table, and a summary under `outputs\mevid_two_camera`. Ground-truth IDs are kept separate from predicted application IDs and never passed to detection or matching. MEVID's released labels are tracklet/crop metadata; the utility does not claim that crop annotations automatically map one-to-one to original video frames. The video URL list and video download instructions are linked in the official repository.

The crop packages are large (the official page lists 30.5 GB for train and 13 GB for test). Download only if you have enough disk space. The preparation tool inventories MEVID tracklets and makes a ground-truth pair table; it does **not** plug MEVID videos into the dashboard or claim a frame-by-frame mapping from cropped tracklets to source video. The app's evaluation uses local IDs from processed uploads, so dataset-based evaluation requires a verified source-clip/frame/tracklet mapping that is not supplied by this utility.

Other candidate check: EPFL's official multi-camera pedestrian sequences are recorded videos with some sparse ground truth, but the listed sequences have only three or four camera streams, below the project's earlier ten-camera dataset constraint. The newer SCOUT release reports 25 views and cross-view person IDs, but its release page describes pre-extracted synchronized frames rather than downloadable original video files; the toolkit points users to the official site for licensing. It therefore is not presented here as a qualifying original-video dataset. See [EPFL sequence details](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/) and [SCOUT dataset page](https://scout.epfl.ch/).

## Output layout

Runs are saved beneath `outputs/review3_results/`:

- Per camera: original-named input and processed MP4, displayed camera name, sampled detection CSV, per-frame track CSV, track summaries with confidence history, retained crop gallery/metadata, rejected-crop reason CSV and bounded preview gallery, statistics, and performance timings.
- Per run: YOLO baseline fingerprints and, when OSNet is installed, separate OSNet fingerprints with model/checkpoint/preprocessing metadata.
- Per run: human-reviewed cross-camera hypotheses in a separate JSON file; downloadable run ZIP includes them.
- Evaluation labels are user-supplied or exported from manually confirmed SAME PERSON hypotheses. No accuracy is shown until valid pairs are available.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover the original detector and crop functions, NMS parameter forwarding and invalid-box filtering, moving/recovered/lost tracks, crop filtering and caps, fingerprint aggregation and missing-candidate handling, dynamic multi-camera ranking and CSV validation, persistent global identity registration and rejection, OSNet adapter CPU/output validation with a mocked Torchreid boundary, and MEVID pair preparation. A synthetic four-camera video test writes and opens full-length MP4s, confirms one ID across three local tracks, and checks that an unrelated no-match track receives no box. The mocked OSNet test does not replace real checkpoint verification or evaluation on manually labelled real-camera footage.

## Known limitations

- The tracker is a lightweight SORT-style Kalman/Hungarian tracker, not ByteTrack. It uses motion and box overlap only; it does not use clothing or gait for local tracking. Crossings and long occlusions can cause lost tracks or ID switches; unmatched people receive a new local ID rather than a risky appearance-only merge.
- OSNet was pretrained on a public Re-ID dataset, not fine-tuned for night cameras. Night-time appearance can still rank poorly.
- Similarity alone never assigns a cross-camera match. A global ID is created for a selected source track, and only manually confirmed tracks are added to it. Each camera's local IDs remain unchanged. Conflicting manual DIFFERENT PEOPLE decisions block video generation until they are resolved.
- Real user videos and user-provided labels are needed for project-specific matching accuracy. A dataset pair table alone is not a result for the dashboard's predictions.
- Processing is sequential, CPU-only, and includes decoding and video writing. Video duration, resolution, image size, and frame interval determine runtime.
