import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

class Config:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'TransUNet_Preprocessed_Data')
    SUBSETS_DIR = os.path.join(ROOT_DIR, 'Subsets')
    CANDIDATES_CSV = os.path.join(ROOT_DIR, 'Common CSV files', 'candidates_V2.csv')
    PATCH_SIZE = 64

def create_spherical_mask(shape, center, radius):
    mask = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
    mask[dist_sq <= radius**2] = 1
    return mask

if __name__ == '__main__':
    df = pd.read_csv(Config.CANDIDATES_CSV)
    for subset_path in glob(os.path.join(Config.SUBSETS_DIR, "subset*")):
        subset_name = os.path.basename(subset_path)
        img_out = os.path.join(Config.OUTPUT_DIR, f'trans_pre_{subset_name}', 'images')
        mask_out = os.path.join(Config.OUTPUT_DIR, f'trans_pre_{subset_name}', 'masks')
        os.makedirs(img_out, exist_ok=True); os.makedirs(mask_out, exist_ok=True)

        subset_ids = [os.path.basename(p).replace(".mhd", "") for p in glob(os.path.join(subset_path, "*.mhd"))]
        subset_df = df[df['seriesuid'].isin(subset_ids)]
        pos = subset_df[subset_df['class'] == 1]
        neg = subset_df[subset_df['class'] == 0].sample(len(pos), random_state=42) if len(subset_df[subset_df['class'] == 0]) > len(pos) else subset_df[subset_df['class'] == 0]
        final_df = pd.concat([pos, neg])

        for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc=f"Preprocessing {subset_name}"):
            mhd_path = os.path.join(subset_path, f"{row['seriesuid']}.mhd")
            itk_img = sitk.ReadImage(mhd_path)
            img_array = sitk.GetArrayFromImage(itk_img)
            origin, spacing = np.array(itk_img.GetOrigin()), np.array(itk_img.GetSpacing())
            v = np.round(np.abs(np.array([row['coordX'], row['coordY'], row['coordZ']]) - origin) / spacing).astype(int)
            vz, vy, vx = v[2], v[1], v[0]
            h = Config.PATCH_SIZE // 2
            if vz-h < 0 or vz+h > img_array.shape[0]: continue
            patch = img_array[vz-h:vz+h, vy-h:vy+h, vx-h:vx+h]
            if patch.shape != (64, 64, 64): continue
            patch = (np.clip(patch, -1000, 400) + 1000) / 1400
            prefix = "pos" if row['class'] == 1 else "neg"
            fname = f"{prefix}_{row['seriesuid']}_{_}.npy"
            np.save(os.path.join(img_out, fname), patch.astype(np.float32))
            mask = np.zeros((64, 64), dtype=np.float32)
            if row['class'] == 1:
                mask_3d = create_spherical_mask((64,64,64), (32,32,32), 5)
                mask = mask_3d[32, :, :].astype(np.float32)
            np.save(os.path.join(mask_out, fname), mask)
    print("✅ TransUNet Preprocessing Done!")