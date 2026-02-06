import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve
)

# --- Configuration ---
class EvalConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'ResNet_Preprocessed_Data')
    MODEL_PATH = os.path.join(ROOT_DIR, 'simpleCNN_unet_best.pth') # Load the U-Net version
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    SEED = 42
    SLICES = 16

config = EvalConfig()

# --- Advanced Simple-UNet Model (Must match training script exactly) ---
class SimpleMultiTaskCNN(nn.Module):
    def __init__(self, in_channels=16):
        super(SimpleMultiTaskCNN, self).__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = self._conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._conv_block(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = self._conv_block(64, 32)
        
        self.seg_final = nn.Conv2d(32, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_fc = nn.Linear(256, 1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        d1 = self.dec1(torch.cat([self.up1(s4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s1], dim=1))
        return self.seg_final(d3), self.clf_fc(self.avgpool(s4).view(s4.size(0), -1))

# --- Dataset Class ---
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.slice_offset = (64 - config.SLICES) // 2
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        patch = np.load(img_path)
        mask_path = img_path.replace('images', 'masks')
        mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((64, 64), dtype=np.float32)
        img_slices = patch[self.slice_offset : self.slice_offset + config.SLICES, :, :]
        if len(mask.shape) == 3: mask = mask[32, :, :] 
        return torch.from_numpy(img_slices).float(), torch.from_numpy(mask).float().unsqueeze(0), torch.tensor(label, dtype=torch.float32)

if __name__ == '__main__':
    print("🔍 Initializing Simple-UNet Evaluation...")
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.seed(config.SEED); random.shuffle(all_files)
    val_files = all_files[int(len(all_files) * (1 - config.VAL_SPLIT)):]
    val_loader = DataLoader(MultiTaskDataset(val_files), batch_size=config.BATCH_SIZE, shuffle=False)

    model = SimpleMultiTaskCNN(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    y_true_clf, y_prob_clf, dice_scores = [], [], []

    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc="Evaluating"):
            imgs, masks, labels = imgs.to(config.DEVICE), masks.to(config.DEVICE), labels.to(config.DEVICE)
            pred_masks, pred_clfs = model(imgs)
            y_prob_clf.extend(torch.sigmoid(pred_clfs).cpu().numpy())
            y_true_clf.extend(labels.cpu().numpy())
            
            # Dice calculation for positive cases
            p_mask = (torch.sigmoid(pred_masks) > 0.5).float()
            for i in range(len(labels)):
                if labels[i] == 1:
                    inter = (p_mask[i] * masks[i]).sum()
                    dice = (2. * inter + 1e-6) / (p_mask[i].sum() + masks[i].sum() + 1e-6)
                    dice_scores.append(dice.item())

    y_prob_clf, y_true_clf = np.array(y_prob_clf).flatten(), np.array(y_true_clf).flatten()
    y_pred_clf = (y_prob_clf > 0.5).astype(int)

    print("\n" + "="*50)
    print("      SIMPLE-UNET MULTI-TASK FINAL REPORT")
    print("="*50)
    print(f"Mean Segmentation Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Area Under ROC Curve (AUC):   {roc_auc_score(y_true_clf, y_prob_clf):.4f}")
    print("-" * 50)
    print(classification_report(y_true_clf, y_pred_clf, target_names=['Healthy', 'Nodule']))

    # Confusion Matrix Visual
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true_clf, y_pred_clf), annot=True, fmt='d', cmap='Oranges')
    plt.title('Simple-UNet Confusion Matrix'); plt.show()