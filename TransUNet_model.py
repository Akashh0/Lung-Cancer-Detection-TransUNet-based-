import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
class TransUNetConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    # Using ResNet preprocessed data structure as requested
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'ResNet_Preprocessed_Data')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'transunet_multitask_best.pth')
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 12  # Adjusted for RTX 4050 VRAM
    EPOCHS = 40
    LEARNING_RATE = 1e-4
    SLICES = 16 
    SEED = 42

config = TransUNetConfig()

# =============================================================================
# 2. MULTI-TASK DATASET
# =============================================================================
class MultiTaskDataset(Dataset):
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

        return (torch.from_numpy(img_slices).float(), 
                torch.from_numpy(mask).float().unsqueeze(0), 
                torch.tensor(label, dtype=torch.float32))

# =============================================================================
# 3. TRANSUNET ARCHITECTURE
# =============================================================================


class TransformerBottleneck(nn.Module):
    def __init__(self, dim, nhead=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, dim)) # 8x8 = 64 patches

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, 64, C]
        x = x + self.pos_embedding
        x = self.transformer(x)
        return x.transpose(1, 2).reshape(b, c, h, w)

class TransUNet(nn.Module):
    def __init__(self, in_channels=16):
        super(TransUNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        
        # Transformer Bottleneck (at 8x8 resolution)
        self.bottleneck = nn.Sequential(nn.MaxPool2d(2), TransformerBottleneck(256))

        # --- DECODER (Corrected Channels for Concatenation) ---
        
        # d1: upsampled bottleneck (128) + s3 skip (256) = 384 input channels
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128))
        
        # d2: upsampled d1 (64) + s2 skip (128) = 192 input channels
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64))
        
        # d3: upsampled d2 (32) + s1 skip (64) = 96 input channels
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32))
        
        self.seg_final = nn.Conv2d(32, 1, kernel_size=1)
        
        # Classification Head (Uses global features from bottleneck)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_fc = nn.Linear(256, 1)

    def forward(self, x):
        # CNN Encoder
        s1 = self.enc1(x)                # [Batch, 64, 64, 64]
        s2 = self.enc2(s1)               # [Batch, 128, 32, 32]
        s3 = self.enc3(s2)               # [Batch, 256, 16, 16]
        
        # Transformer Bottleneck
        bn = self.bottleneck(s3)         # [Batch, 256, 8, 8]
        
        # Segmentation path with Skip Connections
        # 1. Bottleneck to 16x16
        up_1 = self.up1(bn)              # [Batch, 128, 16, 16]
        d1 = self.dec1(torch.cat([up_1, s3], dim=1)) # 128 + 256 = 384
        
        # 2. d1 to 32x32
        up_2 = self.up2(d1)              # [Batch, 64, 32, 32]
        d2 = self.dec2(torch.cat([up_2, s2], dim=1)) # 64 + 128 = 192
        
        # 3. d2 to 64x64
        up_3 = self.up3(d2)              # [Batch, 32, 64, 64]
        d3 = self.dec3(torch.cat([up_3, s1], dim=1)) # 32 + 64 = 96
        
        mask_out = self.seg_final(d3)
        
        # Classification path
        c = torch.flatten(self.avgpool(bn), 1)
        label_out = self.clf_fc(c)
        
        return mask_out, label_out

# =============================================================================
# 4. LOSS & TRAINING
# =============================================================================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

if __name__ == '__main__':
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.shuffle(all_files)

    split = int(len(all_files) * 0.8)
    train_loader = DataLoader(MultiTaskDataset(all_files[:split]), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(all_files[split:]), batch_size=config.BATCH_SIZE)

    model = TransUNet(in_channels=config.SLICES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    clf_criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')

    print(f"🚀 Training TransUNet on {config.DEVICE}...")
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for img, mask, label in loop:
            img, mask, label = img.to(config.DEVICE), mask.to(config.DEVICE), label.to(config.DEVICE)
            optimizer.zero_grad()
            p_mask, p_label = model(img)
            loss = dice_loss(p_mask, mask) + clf_criterion(p_label, label.unsqueeze(1))
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"⭐ Best Model Saved (Loss: {avg_loss:.4f})")