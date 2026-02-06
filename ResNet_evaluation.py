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
    roc_auc_score, roc_curve, precision_recall_curve
)

# --- Configuration (Must match training script) ---
class EvalConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'ResNet_Preprocessed_Data')
    MODEL_PATH = os.path.join(ROOT_DIR, 'resnet_multitask_best.pth')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    SEED = 42
    SLICES = 16

config = EvalConfig()

# --- Helper: Metric Functions ---
def calculate_pixel_dice(pred_mask, true_mask, smooth=1e-6):
    """Calculates Dice Score for the segmentation masks."""
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    intersection = (pred_mask * true_mask).sum()
    dice = (2. * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)
    return dice.item()

# --- Reuse Model & Dataset Classes (Internal) ---
# [The MultiTaskDataset and ResNetMultiTaskModel classes must be present or imported]
# (Included here for a standalone script)

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

class ResNetMultiTaskModel(nn.Module):
    # [Exactly as defined in your training script]
    def __init__(self, in_channels=16):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool = resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 2, 2), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(32, 16, 2, 2), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.seg_final = nn.Conv2d(16, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_fc = nn.Linear(512, 1)
    def forward(self, x):
        x = self.relu(self.bn1(self.first_conv(x))); x = self.maxpool(x)
        l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        d = self.up1(l4); d = self.up2(d); d = self.up3(d); d = self.up4(d); d = self.up5(d)
        return self.seg_final(d), self.clf_fc(self.avgpool(l4).view(l4.size(0), -1))

if __name__ == '__main__':
    print("🔍 Initializing Multi-Task Evaluation...")
    
    # 1. Prepare Data
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.seed(config.SEED); random.shuffle(all_files)
    val_files = all_files[int(len(all_files) * (1 - config.VAL_SPLIT)):]
    val_loader = DataLoader(MultiTaskDataset(val_files), batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Load Model
    model = ResNetMultiTaskModel(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    # 3. Collect Results
    y_true_clf, y_prob_clf = [], []
    dice_scores = []

    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc="Evaluating"):
            imgs, masks, labels = imgs.to(config.DEVICE), masks.to(config.DEVICE), labels.to(config.DEVICE)
            pred_masks, pred_clfs = model(imgs)
            
            # Classification data
            y_prob_clf.extend(torch.sigmoid(pred_clfs).cpu().numpy())
            y_true_clf.extend(labels.cpu().numpy())
            
            # Segmentation data (only for positive samples)
            for i in range(len(labels)):
                if labels[i] == 1:
                    dice = calculate_pixel_dice(pred_masks[i], masks[i])
                    dice_scores.append(dice)

    y_prob_clf = np.array(y_prob_clf).flatten()
    y_true_clf = np.array(y_true_clf).flatten()
    y_pred_clf = (y_prob_clf > 0.5).astype(int)

    # 4. Generate Performance Report
    print("\n" + "="*60)
    print("          LUNG CANCER MULTI-TASK PERFORMANCE REPORT")
    print("="*60)
    print(f"Overall Classification Accuracy: {(y_pred_clf == y_true_clf).mean():.4f}")
    print(f"Mean Segmentation Dice Score:  {np.mean(dice_scores):.4f}")
    print(f"Area Under ROC Curve (AUC):    {roc_auc_score(y_true_clf, y_prob_clf):.4f}")
    print("-" * 60)
    print(classification_report(y_true_clf, y_pred_clf, target_names=['Healthy', 'Nodule']))

    # 5. Visualizations
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_clf, y_pred_clf)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Classification Confusion Matrix')
    ax[0].set_xlabel('Predicted'); ax[0].set_ylabel('Actual')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_clf, y_prob_clf)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_true_clf, y_prob_clf):.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_title('Receiver Operating Characteristic (ROC)')
    ax[1].set_xlabel('False Positive Rate'); ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()