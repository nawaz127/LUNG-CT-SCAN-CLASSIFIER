# app/streamlit_app.py
# =============================================================================
# Lung CT — 3-Model Compare + Grad-CAM (ResNet50 / ViT-B/16 / ResViT)
# - Single model or side-by-side compare
# - ResViT Grad-CAM toggle: CNN last conv  ↔  ViT last block
# - Temperature-aware checkpoints supported
# - Class-label order handling & reordering
# - Download buttons for Grad-CAM overlays (PNG)
# =============================================================================

import os, sys, json, tempfile
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

# Allow "src/..." imports when running via `streamlit run app/streamlit_app.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.resnet import build_resnet
from src.models.vit import build_vit
from src.models.resvit import ResViTClassifier
from src.utils.gradcam_utils import get_cam, overlay_cam

st.set_page_config(page_title="Lung CT — 3-Model Grad-CAM", layout="wide")
st.title("Lung CT Classifier — 3-Model Compare + Grad-CAM")
st.caption("Normal / Benign / Malignant — research demo (not for clinical use)")

# ---------------- Label order handling ----------------
DEFAULT_TRAINED_ORDER = ["Benign", "Malignant", "Normal"]

disp_choice = st.selectbox(
    "Display label order",
    ["Normal, Benign, Malignant", "Benign, Malignant, Normal"],
    index=0
)
DISPLAY_LABELS = [s.strip() for s in disp_choice.split(",")]

st.markdown(
    "Optionally provide an **evaluation.json** (from any experiment folder) "
    "so I can auto-detect the *trained* class order."
)
cls_file = st.file_uploader("Upload evaluation.json (optional)", type=["json"], key="classes_json")

TRAINED_ORDER = DEFAULT_TRAINED_ORDER
if cls_file is not None:
    try:
        d = json.load(cls_file)
        maybe = d.get("classes", d)
        if isinstance(maybe, list) and len(maybe) == 3 and all(isinstance(x, str) for x in maybe):
            TRAINED_ORDER = maybe
            st.success(f"Detected trained order: {TRAINED_ORDER}")
        else:
            st.warning("JSON didn't include a valid `classes` list; using default trained order.")
    except Exception as e:
        st.warning(f"Could not parse JSON: {e}. Using default trained order.")

def reorder_prob(prob: np.ndarray, trained_order, display_order):
    idx = [trained_order.index(cls) for cls in display_order]
    return prob[idx]

# ---------------- UI: single vs compare ----------------
mode = st.radio("Mode", ["Single model", "Compare 3 models"], horizontal=True)

with st.expander("Model checkpoints", expanded=True):
    if mode == "Single model":
        c0, c1 = st.columns(2)
        model_name = c0.selectbox("Model", ["ResNet50", "ViT-B/16", "ResViT"], index=0)
        ckpt_file  = c1.file_uploader("Upload checkpoint (.pt)", type=["pt"], key="single_ckpt")
        resvit_cam_mode = None
        if model_name == "ResViT":
            resvit_cam_mode = st.radio(
                "ResViT Grad-CAM source",
                ["CNN last conv (recommended)", "ViT last block"],
                horizontal=True, index=0
            )
    else:
        c = st.columns(3)
        ckpt_resnet = c[0].file_uploader("ResNet50 checkpoint (.pt)", type=["pt"], key="rn")
        ckpt_vit    = c[1].file_uploader("ViT-B/16 checkpoint (.pt)", type=["pt"], key="vit")
        ckpt_resvit = c[2].file_uploader("ResViT checkpoint (.pt)", type=["pt"], key="rvt")
        resvit_cam_mode = st.radio(
            "ResViT Grad-CAM source (for ResViT column)",
            ["CNN last conv (recommended)", "ViT last block"],
            horizontal=True, index=0
        )
        do_ensemble = st.checkbox("Show ensemble (average of available models)", value=True)

img_file = st.file_uploader("Upload a CT slice (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ---------------- helpers ----------------
def preprocess_pil(img: Image.Image, size: int = 224):
    img224 = img.convert("RGB").resize((size, size))
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])
    x = tf(img224).unsqueeze(0)
    return img224, x

def vit_last_block(m):
    if hasattr(m, "encoder"):
        last = m.encoder.layers[-1]
        return getattr(last, "ln_1", last)
    return m.transformer.encoder.layers[-1]

def load_ckpt_from_uploader(uploader):
    """Return (state_dict, temperature or None). Supports temperature-aware payloads."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploader.read()); tmp.flush()
        loaded = torch.load(tmp.name, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded and "temperature" in loaded:
        return loaded["state_dict"], float(loaded["temperature"])
    return loaded, None

@st.cache_resource
def build_model(model_key: str, num_classes: int):
    if model_key == "resnet":
        m = build_resnet(num_classes, pretrained=False)
        target_layers = [m.layer4[-1]]; is_vit = False
    elif model_key == "vit":
        m = build_vit(num_classes, pretrained=False)
        target_layers = [vit_last_block(m)]; is_vit = True
    else:  # resvit
        m = ResViTClassifier(num_classes=num_classes, pretrained=False)
        target_layers = [m.backbone.res4]; is_vit = False
    return m, target_layers, is_vit

def pick_resvit_targets(m, mode_text):
    if (mode_text or "").startswith("ViT"):
        return [m.backbone.vit_last], True
    return [m.backbone.res4], False

def predict_and_cam(model, x, target_layers, is_vit, temperature=None, pil_for_overlay=None):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if temperature is not None:
            logits = logits / max(float(temperature), 1e-6)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prob = reorder_prob(prob, TRAINED_ORDER, DISPLAY_LABELS)
    cam = get_cam(model, target_layers, is_vit=is_vit)
    grayscale_cam = cam(input_tensor=x)[0]
    overlay = overlay_cam(np.array(pil_for_overlay), grayscale_cam) if pil_for_overlay is not None else None
    return prob, overlay

def to_png_bytes(np_img_uint8: np.ndarray) -> bytes:
    """Convert HxWxC uint8 image to PNG bytes for download."""
    im = Image.fromarray(np_img_uint8.astype(np.uint8))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def model_block(name, uploader, resvit_cam_mode=None, img=None, x=None):
    if uploader is None:
        st.warning(f"Upload a checkpoint for **{name}** to run.")
        return None

    sd, Topt = load_ckpt_from_uploader(uploader)
    key = name.lower()
    model_key = "resnet" if key == "resnet50" else ("vit" if key == "vit-b/16" else "resvit")
    m, target_layers, is_vit = build_model(model_key, num_classes=len(TRAINED_ORDER))
    if model_key == "resvit":
        target_layers, is_vit = pick_resvit_targets(m, resvit_cam_mode)

    m.load_state_dict(sd)
    prob, overlay = predict_and_cam(m, x, target_layers, is_vit, temperature=Topt, pil_for_overlay=img)
    return {"name": name, "prob": prob, "overlay": overlay, "temperature": Topt}

# ---------------- main app logic ----------------
if img_file:
    pil = Image.open(img_file)
    pil224, x = preprocess_pil(pil, size=224)

    if mode == "Single model":
        if ckpt_file is None:
            st.info("Upload a checkpoint to run.")
        else:
            out = model_block(model_name, ckpt_file, resvit_cam_mode, pil224, x)
            if out:
                st.subheader("Prediction")
                st.write({DISPLAY_LABELS[i]: float(out["prob"][i]) for i in range(len(DISPLAY_LABELS))})
                st.bar_chart(pd.DataFrame({"label": DISPLAY_LABELS, "confidence": out["prob"]}).set_index("label"))

                st.image([pil224, out["overlay"]],
                         caption=["Input (224×224)", f"{model_name} Grad-CAM"], width=360)

                # Download button
                if out["overlay"] is not None:
                    png = to_png_bytes(out["overlay"])
                    safe_name = model_name.lower().replace("/", "_").replace(" ", "_")
                    st.download_button(
                        label=f"Download Grad-CAM ({model_name})",
                        data=png,
                        file_name=f"{safe_name}_gradcam.png",
                        mime="image/png"
                    )

    else:
        cols = st.columns(3)
        with cols[0]:
            out_rn = model_block("ResNet50", ckpt_resnet, None, pil224, x)
            if out_rn:
                st.image([pil224, out_rn["overlay"]], caption=["Input", "ResNet Grad-CAM"], width=320)
                st.download_button(
                    "Download Grad-CAM (ResNet50)",
                    data=to_png_bytes(out_rn["overlay"]) if out_rn["overlay"] is not None else b"",
                    file_name="resnet50_gradcam.png",
                    mime="image/png",
                    disabled=out_rn["overlay"] is None
                )
        with cols[1]:
            out_vt = model_block("ViT-B/16", ckpt_vit, None, pil224, x)
            if out_vt:
                st.image([pil224, out_vt["overlay"]], caption=["Input", "ViT Grad-CAM"], width=320)
                st.download_button(
                    "Download Grad-CAM (ViT-B16)",
                    data=to_png_bytes(out_vt["overlay"]) if out_vt["overlay"] is not None else b"",
                    file_name="vit_b16_gradcam.png",
                    mime="image/png",
                    disabled=out_vt["overlay"] is None
                )
        with cols[2]:
            out_rv = model_block("ResViT", ckpt_resvit, resvit_cam_mode, pil224, x)
            if out_rv:
                st.image([pil224, out_rv["overlay"]], caption=["Input", "ResViT Grad-CAM"], width=320)
                st.download_button(
                    f"Download Grad-CAM (ResViT — {resvit_cam_mode or 'CNN'})",
                    data=to_png_bytes(out_rv["overlay"]) if out_rv["overlay"] is not None else b"",
                    file_name=f"resvit_gradcam_{'vit' if (resvit_cam_mode or '').startswith('ViT') else 'cnn'}.png",
                    mime="image/png",
                    disabled=out_rv["overlay"] is None
                )

        st.markdown("### Predictions")
        rows = []
        for out in [out_rn, out_vt, out_rv]:
            if not out: continue
            row = {"model": out["name"]}
            row.update({DISPLAY_LABELS[i]: float(out["prob"][i]) for i in range(len(DISPLAY_LABELS))})
            if out["temperature"] is not None:
                row["T"] = round(float(out["temperature"]), 3)
            rows.append(row)
        if rows:
            df = pd.DataFrame(rows).set_index("model")
            st.dataframe(df.style.format({k: "{:.3f}" for k in DISPLAY_LABELS}))

        if 'do_ensemble' in globals() and do_ensemble:
            probs = [out["prob"] for out in [out_rn, out_vt, out_rv] if out]
            if len(probs) >= 2:
                ens = np.mean(np.stack(probs, axis=0), axis=0)
                st.markdown("### Ensemble (average of available models)")
                st.write({DISPLAY_LABELS[i]: float(ens[i]) for i in range(len(DISPLAY_LABELS))})
                st.bar_chart(pd.DataFrame({"label": DISPLAY_LABELS, "confidence": ens}).set_index("label"))
            else:
                st.info("Provide at least two checkpoints to view the ensemble.")
else:
    st.info("Upload a CT image to begin.")

st.caption(f"Trained order used for mapping: {TRAINED_ORDER}  →  displaying as: {DISPLAY_LABELS}")
