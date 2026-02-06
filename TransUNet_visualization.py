import os
import random
import numpy as np
import torch
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from TransUNet_model import UltimateTransUNet, TransUNetConfig, MultiTaskDataset

# =============================================================================
# 1. CORE SYNCHRONIZATION LOGIC
# =============================================================================
def get_synchronized_data(selected_path, config, candidates_df):
    filename = os.path.basename(selected_path)
    parts = filename.replace(".npy", "").split("_")
    series_uid = parts[1]
    row_idx = int(parts[2])
    
    # Get exact world coordinates from CSV
    row = candidates_df.iloc[row_idx]
    world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])

    # Load original MHD to get geometry
    mhd_path = glob(os.path.join(config.ROOT_DIR, 'Subsets', 'subset*', f"{series_uid}.mhd"))[0]
    itk_img = sitk.ReadImage(mhd_path)
    full_img_array = sitk.GetArrayFromImage(itk_img)
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())

    # World to Voxel Conversion: (World - Origin) / Spacing
    voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
    
    # Standardize image for viewing (Lung Window)
    full_img_array = np.clip(full_img_array, -1000, 400)
    
    return full_img_array, voxel_coords, series_uid

# =============================================================================
# 2. STYLE A: THREE-PANEL DASHBOARD (Synchronized)
# =============================================================================
def plot_dashboard_sync(img_full, patch_3d, voxel_coords, confidence):
    vx, vy, vz = voxel_coords[0], voxel_coords[1], voxel_coords[2]
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor='white')
    
    # Panel 1: Full Slice
    axes[0].imshow(img_full[vz, :, :], cmap='gray')
    axes[0].set_title(f"1. Full CT Scan Slice (Z:{vz})", fontsize=12)
    axes[0].axis('off')

    # Panel 2: Exact Location Highlighted (Mathematically Synced)
    axes[1].imshow(img_full[vz, :, :], cmap='gray')
    circle = plt.Circle((vx, vy), 25, color='red', fill=False, lw=2)
    axes[1].add_artist(circle)
    axes[1].set_title("2. Synchronized Nodule Location", fontsize=12)
    axes[1].axis('off')

    # Panel 3: AI Zoomed Analysis
    axes[2].imshow(patch_3d[32, :, :], cmap='gray')
    axes[2].set_title("3. AI Analysis & Prediction", fontsize=12)
    axes[2].axis('off')

    plt.figtext(0.85, 0.08, f"Prediction: Nodule\nConfidence: {confidence:.2f}%", 
                color='red', fontsize=16, fontweight='bold', ha='center')
    plt.show()

# =============================================================================
# 3. STYLE B: DUAL-PANEL RISK REPORT (Black HUD Style)
# =============================================================================

def plot_risk_report_sync(img_full, voxel_coords, confidence):
    vx, vy, vz = voxel_coords[0], voxel_coords[1], voxel_coords[2]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), facecolor='black')
    
    # Left: Raw Scan
    axes[0].imshow(img_full[vz, :, :], cmap='gray')
    axes[0].set_title("RAW SCAN", color='white', fontsize=22, fontweight='bold', pad=25)
    axes[0].axis('off')

    # Right: AI Diagnosis (With Square HUD)
    axes[1].imshow(img_full[vz, :, :], cmap='gray')
    rect = patches.Rectangle((vx-20, vy-20), 40, 40, lw=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title("AI DIAGNOSIS", color='yellow', fontsize=22, fontweight='bold', pad=25)
    axes[1].axis('off')

    # Bottom Risk HUD
    m_color = 'red' if confidence > 50 else '#555555'
    b_color = 'green' if confidence <= 50 else '#555555'
    
    plt.figtext(0.05, 0.12, f"MALIGNANCY RISK: {confidence:.1f}%", color=m_color, fontsize=28, fontweight='bold')
    plt.figtext(0.05, 0.06, f"BENIGN / SAFE: {100-confidence:.1f}%", color=b_color, fontsize=28, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    config = TransUNetConfig()
    candidates_df = pd.read_csv(os.path.join(config.ROOT_DIR, 'Common CSV files', 'candidates_V2.csv'))
    
    # Load model
    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
    model.eval()

    # Pick random positive sample
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    sample_path = random.choice(pos_paths)
    
    # Sync data and run prediction
    img_full, v_coords, uid = get_synchronized_data(sample_path, config, candidates_df)
    patch_3d = np.load(sample_path)
    
    img_input = patch_3d[24:40, :, :] # Standardize input slices
    img_tensor = torch.from_numpy(img_input).float().unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        _, p_clf = model(img_tensor)
        conf = torch.sigmoid(p_clf).item() * 100

    # Display both styles
    print(f"✅ Generating Perfect Sync Reports for: {uid}")
    plot_dashboard_sync(img_full, patch_3d, v_coords, conf)
    plot_risk_report_sync(img_full, v_coords, conf)