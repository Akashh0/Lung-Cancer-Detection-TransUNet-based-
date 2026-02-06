import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from glob import glob
from tqdm import tqdm

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
class ResNetMultiTaskConfig:
    ROOT_DIR = r'I:\Lung Cancer Project (Simple CNN)'
    PREPROCESSED_PATH = os.path.join(ROOT_DIR, 'ResNet_Preprocessed_Data')
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'resnet_multitask_best.pth')
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16 
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    SLICES = 16 
    PATCH_SIZE = 64
    SEED = 42  # <--- Added missing SEED attribute

config = ResNetMultiTaskConfig()

# =============================================================================
# 2. MULTI-TASK DATASET
# =============================================================================
class MultiTaskDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.slice_offset = (64 - config.SLICES) // 2

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        patch = np.load(img_path)
        
        mask_path = img_path.replace('images', 'masks')
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
        else:
            mask = np.zeros((64, 64), dtype=np.float32)

        # Extract central 16 slices
        img_slices = patch[self.slice_offset : self.slice_offset + config.SLICES, :, :]
        
        # Ensure mask is 2D [64, 64]
        if len(mask.shape) == 3:
            mask = mask[32, :, :] 

        img_tensor = torch.from_numpy(img_slices).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) 
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, mask_tensor, label_tensor

# =============================================================================
# 3. RESNET MULTI-TASK MODEL
# =============================================================================
class ResNetMultiTaskModel(nn.Module):
    def __init__(self, in_channels=16):
        super(ResNetMultiTaskModel, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool 
        
        self.layer1 = resnet.layer1 # 16x16
        self.layer2 = resnet.layer2 # 8x8
        self.layer3 = resnet.layer3 # 4x4
        self.layer4 = resnet.layer4 # 2x2

        # Decoder Path: Upsampling from 2x2 to 64x64
        self.up1 = self._upsample_block(512, 256) # 4x4
        self.up2 = self._upsample_block(256, 128) # 8x8
        self.up3 = self._upsample_block(128, 64)  # 16x16
        self.up4 = self._upsample_block(64, 32)   # 32x32
        self.up5 = self._upsample_block(32, 16)   # 64x64
        self.seg_final = nn.Conv2d(16, 1, kernel_size=1)

        # Classification Path
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_fc = nn.Linear(512, 1)

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x = self.relu(self.bn1(self.first_conv(x))) # 32x32
        x = self.maxpool(x) # 16x16
        l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)

        # Segmentation Head
        d1 = self.up1(l4); d2 = self.up2(d1); d3 = self.up3(d2); d4 = self.up4(d3); d5 = self.up5(d4)
        mask_out = self.seg_final(d5)

        # Classification Head
        c = self.avgpool(l4); c = torch.flatten(c, 1)
        label_out = self.clf_fc(c)
        
        return mask_out, label_out

# =============================================================================
# 4. LOSS FUNCTIONS & TRAINING
# =============================================================================
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

if __name__ == '__main__':
    # Fix Randomness
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    # Find Files
    print(f"🛠️ Scanning for files in {config.PREPROCESSED_PATH}...")
    pos_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'pos*.npy'))
    neg_paths = glob(os.path.join(config.PREPROCESSED_PATH, 'resnet_pre_subset*', 'images', 'neg*.npy'))
    all_files = [(f, 1) for f in pos_paths] + [(f, 0) for f in neg_paths]
    
    if not all_files:
        print("❌ ERROR: No preprocessed data found. Check your paths!")
        exit()

    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    
    train_loader = DataLoader(MultiTaskDataset(all_files[:split_idx]), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(MultiTaskDataset(all_files[split_idx:]), batch_size=config.BATCH_SIZE, num_workers=2)

    model = ResNetMultiTaskModel(in_channels=config.SLICES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    clf_criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')

    print(f"🚀 Training Multi-Task ResNet on {config.DEVICE}...")
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for img, mask, label in loop:
            img, mask, label = img.to(config.DEVICE), mask.to(config.DEVICE), label.to(config.DEVICE)
            
            optimizer.zero_grad()
            p_mask, p_label = model(img)
            
            # Combine Dice Loss (Shape) and BCE Loss (Label)
            loss_seg = dice_loss(p_mask, mask)
            loss_clf = clf_criterion(p_label, label.unsqueeze(1))
            
            total_loss = loss_seg + loss_clf
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item())

        # Save Checkpoint
        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"⭐ Best Model Saved! Loss: {avg_loss:.4f}")