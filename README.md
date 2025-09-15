# Satellite-Change-Detection

In this repository, I keep code that I developed for STAR.VISION Aerospace. **satellite_change_detection.py** implements machine learnng change detection algorithms to detect changes in urbanisation in Southern Africa, particulary South Africa (the country).

In short - this code takes in satellite images, before and after, and outputs a predicted mask showing the change from one image to the next. A more detailed explanation follows below, as well as instructions on how to run this code.

Firstly, put training data in the CWD
Your current working directory (CWD) should look like this (LEVIR-CD layout):
.


├─ train/
│  ├─ A/      # before images
│  ├─ B/      # after images
│  └─ label/  # binary masks
├─ val/   (optional but recommended)
│  ├─ A/  B/  label/
├─ test/  (optional)
│  ├─ A/  B/  label/
└─ satellite_change_detection.py


The training data we use is here -> https://gts.ai/dataset-download/levir-cd/?utm_source=chatgpt.com


**Step 1:**
_Install the dependencies._

pip install torch torchvision opencv-python pillow numpy tqdm scikit-image

**Step 2:**
_Generate csvs for the script._

python satellite_change_detection.py make-csvs

This scans ./train, ./val, ./test and writes:
levir_train.csv
levir_val.csv (if val/ exists)
levir_test.csv (if test/ exists)

**Step 3:**
_Training the model._

Default settings train on levir_train.csv and save model.pt in the CWD:

python satellite_change_detection.py train

Useful knobs:
Faster/bigger tiles: --tile 256 (default) or --tile 512
More/less epochs: --epochs 25
Different learning rate: --lr 5e-4
Change loss blend: --alpha 0.6 (BCE weight; rest is Dice)
No internal val split (use your own val/ later): --val-frac 0.0
Example:
python satellite_change_detection.py train \
  --epochs 25 --batch-size 8 --tile 256 --lr 1e-3 --alpha 0.7 --out model.pt

**Step 4:**
_Evaluating on the validation split._

If you have val/, run:

python satellite_change_detection.py eval

(Uses levir_val.csv and model.pt by default.)

Adjust threshold/cleanup if needed:
python satellite_change_detection.py eval --thr 0.55 --min-blob 50

**Step 5:**
_Predict on a single before/after pair and save visuals._

python satellite_change_detection.py predict \
  --before val/A/1234.png --after val/B/1234.png \
  --out pred_1234.png --thr 0.5 --min-blob 50 \
  --overlay overlay_1234.png --overlay-on before --overlay-alpha 0.5 --outline \
  --diff diff_rgb_1234.png \
  --diff-gray diff_gray_1234.png \
  --diff-masked diff_masked_1234.png \
  --heatmap heat_1234.png
  
What you’ll get:
pred_1234.png — binary change mask (0/255)
overlay_1234.png — mask painted on the before image (red, semi-transparent, with contours)
diff_rgb_1234.png — per-channel absolute difference
diff_gray_1234.png — grayscale absolute difference
diff_masked_1234.png — difference image only within predicted change regions
heat_1234.png — probability heatmap (pre-threshold)
If your own images are slightly misaligned, add --align. For large images, add --tile 512 --overlap 32

**Step 6:**
_Predict on batch of images in the csvs._

This runs the model across all rows and writes outputs with the same base filename:
python satellite_change_detection.py predict-batch \
  --csv levir_val.csv \
  --outdir preds_val \
  --thr 0.5 --tile 512 --overlap 32 --min-blob 50 \
  --overlay-dir overlays_val --overlay-on before --overlay-alpha 0.5 --outline \
  --diff-dir diffs_val --diff-gray-dir diffs_gray_val --diff-masked-dir diffs_masked_val \
  --heatmap-dir heats_val
Resulting folders (created if missing): preds_val/, overlays_val/, diffs_val/, diffs_gray_val/, diffs_masked_val/, heats_val/.

**Fixes to common issues:**

All black / all white mask -> sweep --thr (0.3–0.7), check that training loss decreased, confirm your CSV rows actually include positives.
Speckle → raise --min-blob, or slightly increase --thr.
Seams on big images → increase --overlap (e.g., 64).
Shape mismatch error → ensure the paired images are the same width/height (LEVIR-CD is already aligned).


_**app_streamlit.py**_ turns this code that I have written above into a lighwet web application with a functional UI to easily and accessibly run the change detection code.
