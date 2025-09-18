# Lung CT SCAN Classification (Normal / Benign / Malignant)

End‑to‑end pipeline to train, evaluate, **compare ResNet50 / ViT-B/16 / ResViT**, visualize **Grad-CAM**, and deploy with **Streamlit**.

 ⚠️ Research demo only — **not** for clinical use.

## 🗂️ Dataset

Expected layout under data/processed/:

![](data.png)

Need splits from a single pool?
python src\1_split_dataset.py --src data\raw --dst data\processed --val 0.2 --test 0.2 --seed 42


## 📦 Project Layout (key folders)
![](folder.png)


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


### ✅ ViT-B/16 — experiments/exp02_vit/figs/

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


### ✅ ResViT — experiments/exp03_resvit/figs/

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


## 🖥️ Streamlit App

Run:
conda activate torch_gpu
python -m streamlit run app\streamlit_app.py

- Single or 3‑model compare
- **ResViT** CAM toggle: CNN last conv ↔ ViT last block
- Download **Grad‑CAM overlays** as PNG
- Upload an evaluation.json`to auto‑detect trained class order

## 📜 License

MIT (add a LICENSE file if open‑sourcing).
