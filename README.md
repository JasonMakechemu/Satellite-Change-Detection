1. OVERVIEW

---

This project provides tools for detecting changes between pairs of images (a "before" and an "after" image) using an ONNX model that expects a 6-channel input (3 channels from the before image and 3 from the after image).

It includes scripts for training, batch prediction, and a web-based interface.

Programs included:

1. train_model.py  -  Train a 6-channel change detection model
2. predict_batch_onnx.py  -  Command-line batch predictor
3. streamlit_predict_onnx.py  -  Streamlit web application (graphical interface)

All programs support:

* Tiling large images into smaller patches
* Overlapping tile borders for smooth results
* Stitching predictions back into the full image
* Generating binary masks, overlays, heatmaps, and diffs

---

2. TRAIN_MODEL.PY (TRAINING SCRIPT)

---

Purpose:
Train a change detection model that takes before and after images as input and outputs a change mask.

Inputs:

* Dataset of paired images (before and after)
* Corresponding ground truth change masks

Outputs:

* Trained ONNX model file (e.g., model.onnx)
* Optional training logs and checkpoint files

Basic usage:

python train_model.py 
--data_dir dataset/ 
--epochs 50 
--batch_size 4 
--learning_rate 0.001 
--output_model model.onnx

Command-line arguments:
--data_dir       Directory containing image pairs and masks      (required)
--epochs         Number of training epochs                        (optional, default=50)
--batch_size     Batch size for training                           (optional, default=4)
--learning_rate  Learning rate                                     (optional, default=0.001)
--output_model   Path to save trained ONNX model                   (required)

---

3. PREDICT_BATCH_ONNX.PY (COMMAND-LINE TOOL)

---

Purpose:
Run inference on a batch of image pairs listed in a CSV file.

CSV file format: columns for "before" and "after" image paths, or "A" and "B".
Example:
before,after
dataset/img_001_before.png,dataset/img_001_after.png
dataset/img_002_before.png,dataset/img_002_after.png

Run example:
python predict_batch_onnx.py 
--csv levir_val.csv 
--model model.onnx 
--outdir predictions 
--overlay_dir overlays 
--heatmap_dir heatmaps 
--diff_dir diffs 
--tile 512 
--overlap 32 
--thr 0.15

Outputs:

* Binary masks (predicted change regions) in outdir
* Optional overlays (mask on original image) in overlay_dir
* Heatmaps in heatmap_dir
* Pixel-wise differences in diff_dir

---

4. STREAMLIT_PREDICT_ONNX.PY (WEB APPLICATION)

---

Purpose:
Web-based GUI for predicting change masks from a CSV and ONNX model.

Installation:
pip install streamlit onnxruntime opencv-python-headless numpy tqdm

Run:
streamlit run streamlit_predict_onnx.py

UI Steps:

1. Upload CSV with image pairs.
2. Upload ONNX model.
3. Optionally upload a ZIP of images.
4. Adjust tile size, overlap, threshold, overlay options.
5. Click "Run prediction".
6. Download results as ZIP containing masks, overlays, heatmaps, diffs.

Outputs:

* ZIP file containing:

  * Binary masks (*_mask.png)
  * Optional overlays (*_overlay.png)
  * Heatmaps (*_heatmap.png)
  * Diffs (*_diff.png)

---

5. TILE SIZE AND OVERLAP

---

Tile size: dimension of patches fed to the model (e.g., 512).
Tile overlap: pixels overlapping neighboring tiles (e.g., 32) to reduce seams and improve edge predictions.
Recommended overlap: 32–64 pixels.

---

6. GPU ACCELERATION (OPTIONAL)

---

If using onnxruntime-gpu, change the session provider:
providers=["CUDAExecutionProvider", "CPUExecutionProvider"]

---

7. DEPENDENCIES

---

streamlit               Web UI
onnxruntime             ONNX inference
opencv-python           Image processing
numpy                   Arrays
tqdm                    Progress bar
csv                     CSV parsing
zipfile                 Handling image archives
PyTorch / TensorFlow    Model training (depends on train_model.py)

---

8. TROUBLESHOOTING

---

Cannot read <path>: check file paths or include images in a ZIP.
Shape mismatch: ensure before and after images have identical dimensions.
Model run failed: check ONNX model input shape (6 channels expected).
Blank predictions: lower threshold (e.g., 0.10).
Seams/artifacts: increase overlap.

---

9. EXAMPLE WORKFLOW

---

1. Prepare dataset and ground truth masks.
2. Train model using train_model.py.
3. Run predict_batch_onnx.py for batch predictions or streamlit_predict_onnx.py for GUI predictions.
4. Review outputs (masks, overlays, heatmaps, diffs).

---

END OF DOCUMENTATION
