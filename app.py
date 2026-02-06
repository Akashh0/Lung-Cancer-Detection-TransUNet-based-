import streamlit as st
import os
import random
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset

# =============================================================================
# 1. CORE LOGIC (Synchronized Data & Prediction)
# =============================================================================
@st.cache_resource
def load_model():
    config = TransUNetConfig()
    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
    model.eval()
    return model, config

def get_synchronized_data(selected_path, config, candidates_df):
    filename = os.path.basename(selected_path)
    parts = filename.replace(".npy", "").split("_")
    series_uid = parts[1]
    row_idx = int(parts[2])
    
    row = candidates_df.iloc[row_idx]
    world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])

    mhd_path = glob(os.path.join(config.ROOT_DIR, 'Subsets', 'subset*', f"{series_uid}.mhd"))[0]
    itk_img = sitk.ReadImage(mhd_path)
    full_img_array = sitk.GetArrayFromImage(itk_img)
    origin, spacing = np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing())

    voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
    full_img_array = np.clip(full_img_array, -1000, 400)
    
    return full_img_array, voxel_coords, series_uid

# =============================================================================
# 2. STREAMLIT UI LAYOUT
# =============================================================================
st.set_page_config(page_title="TransUNet AI Diagnostic", layout="wide")
st.title("🫁 AI Lung Cancer Diagnostic Suite")
st.markdown("---")

# Load Resources
model, config = load_model()
candidates_df = pd.read_csv(os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv'))

# Create Export Directory
export_dir = os.path.join(config.ROOT_DIR, "Results_Export")
os.makedirs(export_dir, exist_ok=True)

# Sidebar Controls
st.sidebar.header("Control Panel")
if st.sidebar.button("Pick Random Patient"):
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    st.session_state.sample_path = random.choice(pos_paths)

# Main Execution
if 'sample_path' in st.session_state:
    img_full, v_coords, uid = get_synchronized_data(st.session_state.sample_path, config, candidates_df)
    patch_3d = np.load(st.session_state.sample_path)
    vx, vy, vz = v_coords[0], v_coords[1], v_coords[2]

    # Run Prediction
    img_input = patch_3d[24:40, :, :]
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        _, p_clf = model(img_tensor)
        conf = torch.sigmoid(p_clf).item() * 100

    # Display Info
    st.subheader(f"Patient UID: {uid}")
    
    # Generate Plots
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Raw Scan vs AI Diagnosis")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='black')
        axes[0].imshow(img_full[vz, :, :], cmap='gray')
        axes[0].set_title("RAW SCAN", color='white')
        axes[0].axis('off')
        
        axes[1].imshow(img_full[vz, :, :], cmap='gray')
        rect = patches.Rectangle((vx-25, vy-25), 50, 50, lw=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        axes[1].set_title("AI DETECTION", color='yellow')
        axes[1].axis('off')
        st.pyplot(fig)

    with col2:
        st.write("### Malignancy Assessment")
        risk_val = conf / 100
        st.progress(risk_val)
        if conf > 50:
            st.error(f"MALIGNANCY RISK: {conf:.2f}%")
        else:
            st.success(f"BENIGN PROBABILITY: {100-conf:.2f}%")
            
        # Detailed Patch View
        fig_patch, ax_p = plt.subplots()
        ax_p.imshow(patch_3d[32, :, :], cmap='bone')
        ax_p.set_title("AI Patch Analysis (64x64 Zoomed)")
        ax_p.axis('off')
        st.pyplot(fig_patch)

    # AUTO-SAVE FEATURE
    save_path = os.path.join(export_dir, f"Report_{uid}.png")
    fig.savefig(save_path, facecolor='black')
    st.sidebar.success(f"Report saved to: {export_dir}")

else:
    st.info("Please click 'Pick Random Patient' in the sidebar to start.")