import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
class Config:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    # Separate output folder for SimpleCNN to keep data organized
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'SimpleCNN_Preprocessed_Data')
    SUBSETS_DIR = os.path.join(ROOT_DIR, 'Subsets')
    CANDIDATES_CSV = os.path.join(ROOT_DIR, 'Common CSV files', 'candidates_V2.csv')
    PATCH_SIZE = 64

# --- Helper: Generate Mask ---
def create_spherical_mask(shape, center, radius):
    mask = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    mask[dist_sq <= radius**2] = 1
    return mask

# =============================================================================
# 2. MAIN PREPROCESSING LOGIC
# =============================================================================
if __name__ == '__main__':
    # Load candidates
    df = pd.read_csv(Config.CANDIDATES_CSV)
    
    # Iterate through each subset folder
    for subset_path in glob(os.path.join(Config.SUBSETS_DIR, "subset*")):
        subset_name = os.path.basename(subset_path)
        
        # Structure: SimpleCNN_Preprocessed_Data/simple_pre_subsetX/images/
        img_out = os.path.join(Config.OUTPUT_DIR, f'simple_pre_{subset_name}', 'images')
        mask_out = os.path.join(Config.OUTPUT_DIR, f'simple_pre_{subset_name}', 'masks')
        
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(mask_out, exist_ok=True)

        # Filter CSV for patients in this subset
        subset_ids = [os.path.basename(p).replace(".mhd", "") for p in glob(os.path.join(subset_path, "*.mhd"))]
        subset_df = df[df['seriesuid'].isin(subset_ids)]
        
        # Class Balancing: Match number of negatives to number of positives
        pos = subset_df[subset_df['class'] == 1]
        neg = subset_df[subset_df['class'] == 0]
        if len(neg) > len(pos):
            neg = neg.sample(len(pos), random_state=42)
            
        final_df = pd.concat([pos, neg])

        # Process each nodule candidate
        for idx, row in tqdm(final_df.iterrows(), total=len(final_df), desc=f"Preprocessing {subset_name}"):
            mhd_path = os.path.join(subset_path, f"{row['seriesuid']}.mhd")
            
            # Load CT Image
            itk_img = sitk.ReadImage(mhd_path)
            img_array = sitk.GetArrayFromImage(itk_img)
            origin = np.array(itk_img.GetOrigin())
            spacing = np.array(itk_img.GetSpacing())

            # World Coordinates to Voxel Coordinates
            world_coords = np.array([row['coordX'], row['coordY'], row['coordZ']])
            voxel_coords = np.round(np.abs(world_coords - origin) / spacing).astype(int)
            vz, vy, vx = voxel_coords[2], voxel_coords[1], voxel_coords[0]
            
            h = Config.PATCH_SIZE // 2

            # Boundary Check
            if vz-h < 0 or vz+h > img_array.shape[0] or \
               vy-h < 0 or vy+h > img_array.shape[1] or \
               vx-h < 0 or vx+h > img_array.shape[2]:
                continue
            
            # Extract 3D Patch
            patch = img_array[vz-h:vz+h, vy-h:vy+h, vx-h:vx+h]
            if patch.shape != (64, 64, 64):
                continue
            
            # Intensity Normalization (Clip HU values and scale 0-1)
            patch = (np.clip(patch, -1000, 400) + 1000) / 1400

            # File Naming
            prefix = "pos" if row['class'] == 1 else "neg"
            filename = f"{prefix}_{row['seriesuid']}_{idx}.npy"
            
            # Save Image Patch
            np.save(os.path.join(img_out, filename), patch.astype(np.float32))
            
            # Create & Save 2D Mask (Middle slice of the 3D nodule)
            mask_2d = np.zeros((64, 64), dtype=np.float32)
            if row['class'] == 1:
                # Generate a 5mm radius sphere and take the center slice
                mask_3d = create_spherical_mask((64, 64, 64), (32, 32, 32), 5)
                mask_2d = mask_3d[32, :, :].astype(np.float32)
                
            np.save(os.path.join(mask_out, filename), mask_2d)

    print(f"\n✅ SimpleCNN Preprocessing Finished!")
    print(f"📁 Data saved to: {Config.OUTPUT_DIR}")