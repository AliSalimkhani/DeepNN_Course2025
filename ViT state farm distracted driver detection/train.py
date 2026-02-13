import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.layers import DropPath

BASE_DIR = '/root/.cache/kagglehub/datasets/taojingying11/state-farm-distracted-driver-detection/versions/1'
DATA_DIR = os.path.join(BASE_DIR, 'imgs', 'train')
CSV_PATH = os.path.join(BASE_DIR, 'driver_imgs_list.csv')

IMG_SIZE = 224
BS = 64
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=12),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.35),
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("loading data...")
full_ds = datasets.ImageFolder(DATA_DIR, transform=tfms)
df = pd.read_csv(CSV_PATH)
drivers = df['subject'].unique()
np.random.seed(42)
np.random.shuffle(drivers)

split = int(0.8 * len(drivers))
train_d = list(drivers[:split])
val_d = list(drivers[split:])

mapping = dict(zip(df['img'], df['subject']))
tr_idx, val_idx = [], []

for i, (p, _) in enumerate(full_ds.samples):
    name = os.path.basename(p)
    d = mapping.get(name)
    if d in train_d: tr_idx.append(i)
    elif d in val_d: val_idx.append(i)

train_ds = Subset(full_ds, tr_idx)
val_raw = datasets.ImageFolder(DATA_DIR, transform=val_tfms)
val_ds = Subset(val_raw, val_idx)

train_l = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True)
val_l = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=2.0, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=10, embed_dim=192, depth=8, num_heads=6, mlp_ratio=2.0, drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, drop_rate, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

print("init model")
model = ViT(num_classes=NUM_CLASSES, embed_dim=192, depth=8, num_heads=6, mlp_ratio=2.0, drop_rate=0.2).to(DEVICE)

mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, label_smoothing=0.1, num_classes=NUM_CLASSES)
criterion = SoftTargetCrossEntropy()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

def lr_lambda(epoch):
    if epoch < 5: return (epoch + 1) / 5
    return 0.5 * (1.0 + math.cos(math.pi * (epoch - 5) / (80 - 5)))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda')
best_acc = 0.0
start_epoch = 0

ckpt_path = 'vit_robust_epoch1_acc0.00.pth'
if os.path.exists(ckpt_path):
    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_acc = ckpt['best_val_acc']

for epoch in range(start_epoch, 80):
    model.train()
    t_loss = 0.0
    for imgs, lbls in train_l:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        imgs, lbls = mixup_fn(imgs, lbls)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        t_loss += loss.item()
    
    model.eval()
    v_loss = 0.0
    correct = 0
    total = 0
    val_crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, lbls in val_l:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = val_crit(out, lbls)
            v_loss += loss.item()
            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
    
    scheduler.step()
    acc = 100.0 * correct / total
    print(f"E {epoch+1} | Loss: {t_loss/len(train_l):.3f} | Val: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_acc': best_acc,
        }, f"vit_robust_epoch{epoch+1}_acc{acc:.2f}.pth")

print("done")