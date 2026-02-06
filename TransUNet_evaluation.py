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

# Import the updated architecture and config from your model file
from TransUNet_model import UltimateTransUNet, MultiTaskDataset, TransUNetConfig

def calculate_dice(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return dice.item()

if __name__ == '__main__':
    config = TransUNetConfig()
    print("🔍 Initializing Ultimate TransUNet Evaluation...")
    
    # 1. Load Files from the TransUNet specific directory
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    
    random.seed(config.SEED)
    random.shuffle(all_files)
    
    # Using the same 80/20 split as the training script
    val_files = all_files[int(len(all_files) * 0.8):]
    val_loader = DataLoader(MultiTaskDataset(val_files), batch_size=config.BATCH_SIZE, shuffle=False)

    # 2. Initialize and Load Model
    # Note: Using UltimateTransUNet to match your new training script
    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    
    # Load the "ULTIMATE" weights
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=True))
        model.eval()
        print(f"✅ Loaded model weights from: {config.MODEL_SAVE_PATH}")
    else:
        print(f"❌ Error: Model file not found at {config.MODEL_SAVE_PATH}")
        exit()

    y_true, y_prob, dice_scores = [], [], []

    # 3. Inference Loop
    print(f"🧪 Evaluating on {len(val_files)} samples...")
    with torch.no_grad():
        for imgs, masks, labels in tqdm(val_loader, desc="Testing"):
            imgs, masks, labels = imgs.to(config.DEVICE), masks.to(config.DEVICE), labels.to(config.DEVICE)
            
            p_mask, p_clf = model(imgs)
            
            y_prob.extend(torch.sigmoid(p_clf).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            
            # Calculate Dice for Nodule cases only
            for i in range(len(labels)):
                if labels[i] == 1:
                    dice_scores.append(calculate_dice(p_mask[i], masks[i]))

    y_prob, y_true = np.array(y_prob).flatten(), np.array(y_true).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # 4. Final Performance Report
    print("\n" + "="*60)
    print("      ULTIMATE TRANSUNET MULTI-TASK PERFORMANCE REPORT")
    print("="*60)
    print(f"Mean Segmentation Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Area Under ROC Curve (AUC):   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Overall Accuracy:             {(y_pred == y_true).mean():.4f}")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=['Healthy', 'Nodule']))

    # 5. Visualizations
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma', ax=ax[0],
                xticklabels=['Healthy', 'Nodule'], 
                yticklabels=['Healthy', 'Nodule'])
    ax[0].set_title('Ultimate TransUNet Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax[1].plot(fpr, tpr, color='crimson', lw=2, label=f'AUC = {roc_auc_score(y_true, y_prob):.4f}')
    ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax[1].set_title('ROC Curve')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()