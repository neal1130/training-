#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UNet3D  +  BCE‑Tversky Loss
— 使用與 Swin‑UNet 相同的資料匯入方式：
    from dataset_h5 import HDF5PatchDataset, enumerate_patches_h5, sample_half_foreground_balanced
— 可透過 alpha/beta 強化 FP 懲罰（把 beta 設大於 alpha）
"""

import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ========= 與 Swin‑UNet 一樣的匯入 =========
from dataset_h5 import (
    HDF5PatchDataset,
    enumerate_patches_h5,
    sample_half_foreground_balanced,
)

# 建議：Windows 下多進程讀同一 HDF5 可先關掉 file locking（如不需要可移除）
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# -------------------------------
# UNet3D
# -------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feats=(32, 64, 128, 256)):
        super().__init__()
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        for f in feats:
            self.downs.append(nn.Sequential(
                nn.Conv3d(ch, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(inplace=True),
                nn.Conv3d(f, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(inplace=True),
            ))
            self.pools.append(nn.MaxPool3d(2))
            ch = f
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch*2, 3, padding=1), nn.BatchNorm3d(ch*2), nn.ReLU(inplace=True),
            nn.Conv3d(ch*2, ch*2, 3, padding=1), nn.BatchNorm3d(ch*2), nn.ReLU(inplace=True),
        )
        self.ups, self.upc = nn.ModuleList(), nn.ModuleList()
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose3d(f*2, f, 2, stride=2))
            self.upc.append(nn.Sequential(
                nn.Conv3d(f*2, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(inplace=True),
                nn.Conv3d(f, f, 3, padding=1), nn.BatchNorm3d(f), nn.ReLU(inplace=True),
            ))
        self.final = nn.Conv3d(feats[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x); skips.append(x); x = pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.upc, reversed(skips)):
            x = up(x); x = torch.cat([skip, x], dim=1); x = conv(x)
        return self.final(x)


# -------------------------------
# BCE + Tversky（可重罰 FP）
# alpha 懲罰 FN；beta 懲罰 FP（想重罰 FP → beta > alpha）
# gamma > 1 變成 focal‑tversky
# -------------------------------
class BCEPlusTverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 gamma: float = 1.0,          # 1.0: 一般 Tversky；>1: focal‑tversky
                 lambda_bce: float = 1.0,
                 lambda_tv:  float = 1.0,
                 smooth: float = 1e-5,
                 eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.gamma = gamma
        self.lambda_bce, self.lambda_tv = lambda_bce, lambda_tv
        self.smooth, self.eps = smooth, eps
        self.bce = nn.BCEWithLogitsLoss()
        self.last_bce = None
        self.last_tv  = None
        self.last_ti  = None

    def forward(self, logits, target):
        bce_loss = self.bce(logits, target)

        probs = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        p = probs.view(-1)
        t = target.view(-1)

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()

        ti = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth + self.eps)
        tv_loss = (1.0 - ti) if self.gamma == 1.0 else torch.pow((1.0 - ti), self.gamma)

        self.last_bce = bce_loss.detach()
        self.last_tv  = tv_loss.detach()
        self.last_ti  = ti.detach()
        return self.lambda_bce * bce_loss + self.lambda_tv * tv_loss


# -------------------------------
# collate（與你原本相同語意）
# -------------------------------
def collate_fn(batch):
    imgs, masks = zip(*batch)
    return torch.stack(imgs), torch.stack(masks)


# -------------------------------
# 訓練主程式（與 Swin 的資料介面一致）
# -------------------------------
def train_unet3d_h5(
    h5_path: str,
    patch_size=(48, 48, 16),
    overlap=(8, 8, 8),
    batch_size=20,
    epochs=200,
    patience=10,
    model_path=r'D:\kun\spine\unet3d_best.pth',
    # Tversky 參數：重罰 FP → 調高 beta
    tv_alpha=0.3, tv_beta=0.7, tv_gamma=1.0,
    lambda_bce=1.0, lambda_tv=1.0,
    num_workers: int = 4,
    prefetch_factor: int = 3
):
    # 與 Swin 一致：只在 XY 平面旋轉
    HDF5PatchDataset.plane_axes = [(2, 3)]   # 2=H, 3=W

    # 列舉 + 平衡取樣（直接用你 dataset_h5 的實作）
    all_items = enumerate_patches_h5(h5_path, patch_size, overlap)
    sel_items = sample_half_foreground_balanced(all_items)

    n_val = max(1, int(len(sel_items) * 0.2))
    n_tr  = len(sel_items) - n_val
    tr_i, va_i = random_split(sel_items, [n_tr, n_val])

    ds_tr = HDF5PatchDataset(h5_path, tr_i, patch_size, rotate=True)
    ds_va = HDF5PatchDataset(h5_path, va_i, patch_size, rotate=False)

    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    dl_va = DataLoader(
        ds_va, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    model     = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BCEPlusTverskyLoss(
        alpha=tv_alpha, beta=tv_beta, gamma=tv_gamma,
        lambda_bce=lambda_bce, lambda_tv=lambda_tv
    )

    best, wait = float('inf'), 0
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        tbar = tqdm(dl_tr, desc=f"Epoch {ep}/{epochs} ▶ Train", unit="batch")
        for imgs, masks in tbar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss   = criterion(logits, masks)

            bce_val = float(criterion.last_bce.item())
            tv_val  = float(criterion.last_tv.item())
            ti_val  = float(criterion.last_ti.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tbar.set_postfix(
                loss=f"{loss.item():.4f}",
                bce =f"{bce_val:.4f}",
                tv  =f"{tv_val:.4f}",
                ti  =f"{ti_val:.4f}",
            )

        # ---- Val ----
        model.eval()
        v_loss = v_bce = v_tv = v_ti = 0.0
        with torch.no_grad():
            vbar = tqdm(dl_va, desc=f"Epoch {ep}/{epochs} ▶ Val  ", unit="batch")
            for imgs, masks in vbar:
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                loss   = criterion(logits, masks)

                bce_val = float(criterion.last_bce.item())
                tv_val  = float(criterion.last_tv.item())
                ti_val  = float(criterion.last_ti.item())

                v_loss += loss.item()
                v_bce  += bce_val
                v_tv   += tv_val
                v_ti   += ti_val

                vbar.set_postfix(
                    val_loss=f"{loss.item():.4f}",
                    val_bce =f"{bce_val:.4f}",
                    val_tv  =f"{tv_val:.4f}",
                    val_ti  =f"{ti_val:.4f}",
                )

        n_batches = len(dl_va)
        avg_loss  = v_loss / n_batches
        avg_bce   = v_bce  / n_batches
        avg_tv    = v_tv   / n_batches
        avg_ti    = v_ti   / n_batches

        tqdm.write(f"[{ep}/{epochs}] Val loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, TV: {avg_tv:.4f}, TI: {avg_ti:.4f}")

        if avg_loss < best:
            best, wait = avg_loss, 0
            torch.save(model.state_dict(), model_path)
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"Early stopping @ epoch {ep}")
                break


if __name__ == '__main__':
    # Windows DataLoader 建議使用 spawn
    torch.multiprocessing.set_start_method('spawn', force=True)

    train_unet3d_h5(
        h5_path    = r'D:\kun\spine\rawd_ataset.h5',
        patch_size = (64, 64, 16),
        overlap    = (8, 8, 8),
        batch_size = 20,
        epochs     = 1000,
        patience   = 15,
        model_path = r'D:\kun\spine\unet3d_best.pth',
        # 想更重罰 FP → 提高 beta；若要 focal‑tversky，將 tv_gamma>1
        tv_alpha=0.3, tv_beta=0.7, tv_gamma=1.0,
        lambda_bce=1.0, lambda_tv=1.0,
        num_workers=4, prefetch_factor=3,
    )
