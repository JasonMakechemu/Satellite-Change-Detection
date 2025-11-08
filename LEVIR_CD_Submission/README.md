Project: LEVIR Change Detection ONNX Submission

1. ONNX Model
--------------
Location: onnx_model/model.onnx
Description:
  - Exported from PyTorch (Siamese U-Net) for change detection.
  - Opset version < 13 to ensure compatibility with ONNX runtime.
Usage:
  - Loaded in inference.py for running predictions on image pairs.

2. Python Inference Script
---------------------------
Location: inference/inference.py
Description:
  - Reads test images (before/after pairs).
  - Supports sliding-window tiling for large images.
  - Runs ONNX inference to generate binary masks.
  - Produces optional visualizations: overlays, heatmaps, RGB/gray differences.
Dependencies (listed in requirements.txt):
  - onnxruntime
  - numpy
  - opencv-python
  - pillow
  - tqdm
Example Run:
  python inference.py \
      --onnx_model ../onnx_model/model.onnx \
      --csv ../test_data/test.csv \
      --outdir ../outputs/preds_val \
      --overlay-dir ../outputs/overlays_val \
      --heatmap-dir ../outputs/heats_val \
      --diff-dir ../outputs/diffs_val \
      --diff-gray-dir ../outputs/diffs_gray_val \
      --diff-masked-dir ../outputs/diffs_masked_val \
      --tile 512 --overlap 32 --thr 0.15 --min-blob 50

3. Test Data
-------------
Location: test_data/
Contents:
  - A/ : Before images
  - B/ : After images
  - label/ : Ground-truth masks (for optional accuracy verification)
Notes:
  - Used to validate the inference script and generate sample outputs.

4. Outputs
-----------
Location: outputs/
Description:
  - preds_val/        : Binary masks (0/255) representing predicted changes.
  - overlays_val/     : “Before” images overlaid with predicted changes in red.
  - heats_val/        : Probability heatmaps of predicted changes.
  - diffs_val/        : Absolute difference images in RGB.
  - diffs_gray_val/   : Absolute difference images in grayscale.
  - diffs_masked_val/ : RGB absolute difference masked by predicted changes.

5. How to Run
--------------
1) Set up the Python environment:
   cd inference
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Run batch inference:
   python inference.py \
       --onnx_model ../onnx_model/model.onnx \
       --csv ../test_data/test.csv \
       --outdir ../outputs/preds_val \
       --overlay-dir ../outputs/overlays_val \
       --heatmap-dir ../outputs/heats_val \
       --diff-dir ../outputs/diffs_val \
       --diff-gray-dir ../outputs/diffs_gray_val \
       --diff-masked-dir ../outputs/diffs_masked_val \
       --tile 512 --overlap 32 --thr 0.15 --min-blob 50

3) Check the outputs in the outputs/ folder.

6. Optional Accuracy Check
---------------------------
- If ground-truth labels are included, the script can compute IoU or other metrics to validate predictions.
- Use test_data/label/ images as reference.

7. Packaging Notes
------------------
- The submission folder should include:
    onnx_model/
    inference/
    test_data/
    outputs/ (optional)
    README.txt (this documentation)
- Zip the entire folder for submission:
    zip -r LEVIR_CD_Submission.zip LEVIR_CD_Submission/
- Reviewer should be able to run the inference script using the ONNX model and reproduce all outputs.



Using Your Own Images for Inference
-----------------------------------

1. Folder Structure:
   - Create a folder for your images with the following subfolders:

     my_images/
         A/      # "Before" images
         B/      # "After" images
         label/  # Optional: ground-truth masks for verification

   - Filenames in A/ and B/ must match exactly (e.g., image_01.png in both folders).
   - Labels are optional but recommended if you want to validate accuracy.

2. Prepare a CSV:
   - The inference script requires a CSV listing all image pairs. Example CSV format:

     before,after,mask
     my_images/A/image_01.png,my_images/B/image_01.png,my_images/label/image_01.png
     my_images/A/image_02.png,my_images/B/image_02.png,my_images/label/image_02.png

3. Run Inference:
   - Use the CSV with the script and specify output directories:

     python inference.py \
         --onnx_model ../onnx_model/model.onnx \
         --csv my_images.csv \
         --outdir ../outputs/preds_my_images \
         --overlay-dir ../outputs/overlays_my_images \
         --heatmap-dir ../outputs/heats_my_images \
         --diff-dir ../outputs/diffs_my_images \
         --diff-gray-dir ../outputs/diffs_gray_my_images \
         --diff-masked-dir ../outputs/diffs_masked_my_images \
         --tile 512 --overlap 32 --thr 0.15 --min-blob 50

4. Check Outputs:
   - Binary masks and visualizations will appear in the specified output folders.
   - Overlays show predicted changes on the "before" images.
   - Heatmaps show the probability of change.
   - Diff images show absolute differences between before and after images.

5. Notes:
   - Input images must be pixel-aligned and of the same size.
   - Use the `--tile` option for large images and `--align` if slight misalignment is present.



Streamlit App for Satellite Change Detection
-----------------------------------

This Streamlit application provides an interactive interface for performing batch satellite image change detection using a pre-trained ONNX model. Users can upload a CSV file containing pairs of before and after images, select an ONNX model file, and configure prediction parameters such as tile size, overlap, probability threshold, and minimum blob size. The app supports sliding-window inference for large images and generates multiple outputs for each image pair, including binary change masks, overlay visualizations, probability heatmaps, RGB and grayscale difference images, and masked difference images. Users can preview all results directly in the browser and optionally save outputs to organized folders for further analysis.

To run the app, first ensure that the required Python packages are installed:

pip install streamlit onnxruntime opencv-python pillow pandas numpy

Then start the app with:

streamlit run streamlit_app/app.py

In the web interface, select the ONNX model, upload your CSV file, adjust the parameters as needed, and view results in real time. Enabling the “Save outputs to folder” option will automatically write all generated files into structured directories for easy access and downstream processing.
