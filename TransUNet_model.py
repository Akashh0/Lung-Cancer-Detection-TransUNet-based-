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
# 1. ULTIMATE CONFIGURATION
# =============================================================================
class TransUNetConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'TransUNet_Preprocessed_Data')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'transunet_ULTIMATE_best.pth')
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 10 
    EPOCHS = 50      
    LEARNING_RATE = 2e-4 
    SLICES = 16 
    SEED = 42

config = TransUNetConfig()

# =============================================================================
# 2. DATASET CLASS (The Missing Piece)
# =============================================================================
class MultiTaskDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        # Ensures we take the middle slices from the 64-slice patch
        self.slice_offset = (64 - config.SLICES) // 2

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        patch = np.load(img_path)
        
        # Locate corresponding mask
        mask_path = img_path.replace('images', 'masks')
        mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((64, 64), dtype=np.float32)

        # 3D to 2D-Multi-Slice conversion
        img_slices = patch[self.slice_offset : self.slice_offset + config.SLICES, :, :]
        
        # Ensure mask is 2D
        if len(mask.shape) == 3: mask = mask[32, :, :] 

        return (torch.from_numpy(img_slices).float(), 
                torch.from_numpy(mask).float().unsqueeze(0), 
                torch.tensor(label, dtype=torch.float32))

# =============================================================================
# 3. TRANSFORMER & MODEL BLOCKS
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class UltimateTransUNet(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        
        # Transformer Bottleneck (8x8)
        self.bottleneck_pool = nn.MaxPool2d(2)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 256))
        self.transformers = nn.Sequential(*[TransformerBlock(256) for _ in range(6)])
        
        # Decoder with Skip Connections
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(192, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))
        
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True))
        
        self.seg_final = nn.Conv2d(32, 1, kernel_size=1)
        self.clf_head = nn.Linear(256, 1)

    def forward(self, x):
        s1 = self.enc1(x); s2 = self.enc2(s1); s3 = self.enc3(s2)
        b = self.bottleneck_pool(s3)
        b_flat = b.flatten(2).transpose(1, 2) + self.pos_embed
        t_out = self.transformers(b_flat)
        b_out = t_out.transpose(1, 2).reshape(b.shape)
        
        d1 = self.dec1(torch.cat([self.up1(b_out), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), s2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), s1], dim=1))
        
        return self.seg_final(d3), self.clf_head(torch.mean(t_out, dim=1))

# =============================================================================
# 4. TRAINING LOGIC
# =============================================================================
def hybrid_loss(p_mask, mask, p_label, label):
    p_mask_sig = torch.sigmoid(p_mask)
    dice = 1 - (2.*(p_mask_sig * mask).sum() + 1e-6) / (p_mask_sig.sum() + mask.sum() + 1e-6)
    bce = nn.functional.binary_cross_entropy_with_logits(p_label, label.unsqueeze(1), reduction='none')
    focal = (0.25 * (1 - torch.exp(-bce))**2 * bce).mean()
    return dice + focal

if __name__ == '__main__':
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'trans_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    
    if not all_files:
        print(f"❌ Error: No files found in {config.PREPROCESSED_PATH}")
        exit()
        
    random.shuffle(all_files)
    split = int(len(all_files) * 0.8)
    train_loader = DataLoader(MultiTaskDataset(all_files[:split]), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(all_files[split:]), batch_size=config.BATCH_SIZE)

    model = UltimateTransUNet(in_channels=config.SLICES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    best_loss = float('inf')
    print(f"🚀 Training ULTIMATE TransUNet on {config.DEVICE}...")
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for img, mask, label in loop:
            img, mask, label = img.to(config.DEVICE), mask.to(config.DEVICE), label.to(config.DEVICE)
            optimizer.zero_grad()
            p_mask, p_label = model(img)
            loss = hybrid_loss(p_mask, mask, p_label, label)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = train_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"⭐ Saved Ultimate Model! Loss: {avg_loss:.4f}")