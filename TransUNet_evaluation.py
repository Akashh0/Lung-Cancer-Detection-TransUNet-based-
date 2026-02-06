import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
import os, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# Import config and model from the training file
from TransUNet_model import TransUNet, MultiTaskDataset, TransUNetConfig

if __name__ == '__main__':
    config = TransUNetConfig()
    print("🔍 Initializing TransUNet Evaluation...")
    
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.seed(config.SEED); random.shuffle(all_files)
    val_files = all_files[int(len(all_files) * 0.8):]
    val_loader = DataLoader(MultiTaskDataset(val_files), batch_size=config.BATCH_SIZE, shuffle=False)

    model = TransUNet(in_channels=config.SLICES).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()

    y_true, y_prob, dice_scores = [], [], []

    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader):
            imgs, masks, labels = imgs.to(config.DEVICE), masks.to(config.DEVICE), labels.to(config.DEVICE)
            p_mask, p_clf = model(imgs)
            y_prob.extend(torch.sigmoid(p_clf).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
            # Dice logic
            p_mask_bin = (torch.sigmoid(p_mask) > 0.5).float()
            for i in range(len(labels)):
                if labels[i] == 1:
                    inter = (p_mask_bin[i] * masks[i]).sum()
                    dice = (2.*inter + 1e-6) / (p_mask_bin[i].sum() + masks[i].sum() + 1e-6)
                    dice_scores.append(dice.item())

    y_prob, y_true = np.array(y_prob).flatten(), np.array(y_true).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    print("\n" + "="*50)
    print("      TRANSUNET MULTI-TASK FINAL REPORT")
    print("="*50)
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"AUC Score:       {roc_auc_score(y_true, y_prob):.4f}")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Nodule']))
    
    # =============================================================================
    # 5. VISUALIZATIONS (Confusion Matrix & ROC Curve)
    # =============================================================================
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0],
                xticklabels=['Healthy', 'Nodule'], 
                yticklabels=['Healthy', 'Nodule'])
    ax[0].set_title('TransUNet Confusion Matrix')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    # ROC Curve for Visual Verification
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc_score(y_true, y_prob):.4f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_title('Receiver Operating Characteristic')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()