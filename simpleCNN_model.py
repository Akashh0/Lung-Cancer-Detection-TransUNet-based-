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
class SimpleCNNConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'ResNet_Preprocessed_Data')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'simpleCNN_unet_best.pth')
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16 
    EPOCHS = 40        # Increased epochs for better convergence
    LEARNING_RATE = 1e-4
    SLICES = 16 
    SEED = 42

config = SimpleCNNConfig()

# =============================================================================
# 2. DATASET
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
# 3. ADVANCED SIMPLE-UNET MODEL (With Skip Connections)
# =============================================================================
class SimpleMultiTaskCNN(nn.Module):
    def __init__(self, in_channels=16):
        super(SimpleMultiTaskCNN, self).__init__()
        
        # --- ENCODER ---
        self.enc1 = self._conv_block(in_channels, 32)   # 64x64
        self.enc2 = self._conv_block(32, 64)           # 32x32
        self.enc3 = self._conv_block(64, 128)          # 16x16
        self.enc4 = self._conv_block(128, 256)         # 8x8 (Bottleneck)
        self.pool = nn.MaxPool2d(2, 2)

        # --- DECODER (Upsampling + Concatenation) ---
        # 8x8 -> 16x16
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(256, 128) # 128 (up1) + 128 (enc3 skip)
        
        # 16x16 -> 32x32
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)  # 64 (up2) + 64 (enc2 skip)
        
        # 32x32 -> 64x64
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(64, 32)   # 32 (up3) + 32 (enc1 skip)
        
        self.seg_final = nn.Conv2d(32, 1, kernel_size=1)

        # --- CLASSIFICATION HEAD ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_fc = nn.Linear(256, 1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder Path
        s1 = self.enc1(x)                # Skip 1 (64x64)
        s2 = self.enc2(self.pool(s1))    # Skip 2 (32x32)
        s3 = self.enc3(self.pool(s2))    # Skip 3 (16x16)
        s4 = self.enc4(self.pool(s3))    # Bottleneck (8x8)

        # Segmentation Path (Decoder with Skips)
        d1 = self.up1(s4)                # 16x16
        d1 = self.dec1(torch.cat([d1, s3], dim=1))
        
        d2 = self.up2(d1)                # 32x32
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        
        d3 = self.up3(d2)                # 64x64
        d3 = self.dec3(torch.cat([d3, s1], dim=1))
        
        mask_out = self.seg_final(d3)

        # Classification Path
        c = self.avgpool(s4)
        c = torch.flatten(c, 1)
        label_out = self.clf_fc(c)

        return mask_out, label_out

# =============================================================================
# 4. TRAINING LOGIC
# =============================================================================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

if __name__ == '__main__':
    random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    
    # Robust File Loading
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    random.shuffle(all_files)

    split = int(len(all_files) * 0.8)
    train_loader = DataLoader(MultiTaskDataset(all_files[:split]), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiTaskDataset(all_files[split:]), batch_size=config.BATCH_SIZE)

    model = SimpleMultiTaskCNN(in_channels=config.SLICES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Scheduler helps fine-tune the model toward the end of training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    clf_criterion = nn.BCEWithLogitsLoss()

    best_loss = float('inf')

    print(f"🚀 Training Simple-UNet on {config.DEVICE}...")
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
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"⭐ Saved Best Simple-UNet Model (Loss: {avg_loss:.4f})")