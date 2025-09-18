# Lung CT SCAN Classification (Normal / Benign / Malignant)

End‑to‑end pipeline to train, evaluate, **compare ResNet50 / ViT-B/16 / ResViT**, visualize **Grad-CAM**, and deploy with **Streamlit**.

> ⚠️ Research demo only — **not** for clinical use.

## 🗂️ Dataset

Expected layout under `data/processed/`:

data/processed/
├─ Benign/
├─ Malignant/
└─ Normal/

Need splits from a single pool?
python src\1_split_dataset.py --src data\raw --dst data\processed --val 0.2 --test 0.2 --seed 42


## 📦 Project Layout (key folders)
E:\Project\lung-ct-3class_FINAL
├─ .streamlit\ # Streamlit server config (upload limits)
├─ app
│ └─ streamlit_app.py # 3-model demo + Grad-CAM + download buttons
├─ data
│ ├─ raw\ # put your original dataset here
│ └─ processed\ # split dataset (train/val/test) goes here
├─ experiments
│ ├─ exp01_resnet
│ ├─ exp02_vit
│ └─ exp03_resvit\ # each contains best.pt, figs/, *.npz, evaluation.json, etc.
├─ src
│ ├─ 1_split_dataset.py
│ ├─ 2_train.py
│ ├─ 3_eval.py
│ ├─ 4_make_gradcam_panel.py
│ ├─ 5_update_readme.py
│ ├─ 6_plot_curves.py
│ ├─ 7_plot_confmat.py # alias: 7_plot_confusion_and_classwise.py
│ ├─ 7_plot_confusion_and_classwise.py # generates confusion_matrix.png + classwise_metrics.md
│ ├─ 8_calibration_and_thresholds.py
│ ├─ 9_temp_scaling_and_calibration_metrics.py
│ ├─ 10_export_calibrated_checkpoint.py
│ ├─ models
│ │ ├─ resnet.py
│ │ ├─ vit.py
│ │ └─ resvit.py # hybrid CNN+ViT with CAM targets (CNN/ViT selectable)
│ └─ utils
│ └─ gradcam_utils.py
├─ pyproject.toml # installable package (editable)
├─ requirements.txt
└─ README.md 


## 🖼️ Figures (existing files under `figs/`)

> Links point to real files in your repo’s `experiments/expXX_*/figs/` folders.  
> These filenames follow your generated class names (e.g., **“Bengin cases”**).

### ✅ ResNet50 — `experiments/exp01_resnet/figs/`

**Confusion Matrix**  
![confusion matrix](experiments/exp01_resnet/figs/confusion_matrix.png)

**Precision-Recall**
![pr_0_Bengin cases](experiments/exp01_resnet/figs/pr_0_Bengin%20cases.png)
![pr_1_Malignant cases](experiments/exp01_resnet/figs/pr_1_Malignant%20cases.png)
![pr_2_Normal cases](experiments/exp01_resnet/figs/pr_2_Normal%20cases.png)

**ROC**
![roc_0_Bengin cases](experiments/exp01_resnet/figs/roc_0_Bengin%20cases.png)
![roc_1_Malignant cases](experiments/exp01_resnet/figs/roc_1_Malignant%20cases.png)
![roc_2_Normal cases](experiments/exp01_resnet/figs/roc_2_Normal%20cases.png)

**Calibration (Reliability)**
![calibration_0_Bengin cases](experiments/exp01_resnet/figs/calibration_0_Bengin%20cases.png)
![calibration_1_Malignant cases](experiments/exp01_resnet/figs/calibration_1_Malignant%20cases.png)
![calibration_2_Normal cases](experiments/exp01_resnet/figs/calibration_2_Normal%20cases.png)


### ✅ ViT-B/16 — `experiments/exp02_vit/figs/`

**Confusion Matrix**  
![confusion matrix](experiments/exp02_vit/figs/confusion_matrix.png)

**Precision-Recall**
![pr_0_Bengin cases](experiments/exp02_vit/figs/pr_0_Bengin%20cases.png)
![pr_1_Malignant cases](experiments/exp02_vit/figs/pr_1_Malignant%20cases.png)
![pr_2_Normal cases](experiments/exp02_vit/figs/pr_2_Normal%20cases.png)

**ROC**
![roc_0_Bengin cases](experiments/exp02_vit/figs/roc_0_Bengin%20cases.png)
![roc_1_Malignant cases](experiments/exp02_vit/figs/roc_1_Malignant%20cases.png)
![roc_2_Normal cases](experiments/exp02_vit/figs/roc_2_Normal%20cases.png)

**Calibration (Reliability)**
![calibration_0_Bengin cases](experiments/exp02_vit/figs/calibration_0_Bengin%20cases.png)
![calibration_1_Malignant cases](experiments/exp02_vit/figs/calibration_1_Malignant%20cases.png)
![calibration_2_Normal cases](experiments/exp02_vit/figs/calibration_2_Normal%20cases.png)


### ✅ ResViT — `experiments/exp03_resvit/figs/`

**Confusion Matrix**  
![confusion matrix](experiments/exp03_resvit/figs/confusion_matrix.png)

**Precision-Recall**
![pr_0_Bengin cases](experiments/exp03_resvit/figs/pr_0_Bengin%20cases.png)
![pr_1_Malignant cases](experiments/exp03_resvit/figs/pr_1_Malignant%20cases.png)
![pr_2_Normal cases](experiments/exp03_resvit/figs/pr_2_Normal%20cases.png)

**ROC**
![roc_0_Bengin cases](experiments/exp03_resvit/figs/roc_0_Bengin%20cases.png)
![roc_1_Malignant cases](experiments/exp03_resvit/figs/roc_1_Malignant%20cases.png)
![roc_2_Normal cases](experiments/exp03_resvit/figs/roc_2_Normal%20cases.png)

**Calibration (Reliability)**
![calibration_0_Bengin cases](experiments/exp03_resvit/figs/calibration_0_Bengin%20cases.png)
![calibration_1_Malignant cases](experiments/exp03_resvit/figs/calibration_1_Malignant%20cases.png)
![calibration_2_Normal cases](experiments/exp03_resvit/figs/calibration_2_Normal%20cases.png)

## 🗂️ Dataset

data/processed/
├─ Benign/
├─ Malignant/
└─ Normal/


## 🖥️ Streamlit App

Run:
conda activate torch_gpu
python -m streamlit run app\streamlit_app.py

- Single or 3‑model compare
- **ResViT** CAM toggle: CNN last conv ↔ ViT last block
- Download **Grad‑CAM overlays** as PNG
- Upload an `evaluation.json` to auto‑detect trained class order


