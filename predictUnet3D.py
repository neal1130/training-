#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
import dask.array as da
from dask.diagnostics import ProgressBar
from tifffile import memmap as tif_memmap, imwrite
from tqdm import tqdm
import torch.nn as nn

# 定義 3D U-Net 模型
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, feats=[32,64,128,256]):
        super().__init__()
        self.downs, self.pools = nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        for f in feats:
            self.downs.append(nn.Sequential(
                nn.Conv3d(ch, f, 3, padding=1),
                nn.BatchNorm3d(f),
                nn.ReLU(inplace=True),
                nn.Conv3d(f, f, 3, padding=1),
                nn.BatchNorm3d(f),
                nn.ReLU(inplace=True),
            ))
            self.pools.append(nn.MaxPool3d(2))
            ch = f
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch*2, 3, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.ReLU(True),
            nn.Conv3d(ch*2, ch*2, 3, padding=1),
            nn.BatchNorm3d(ch*2),
            nn.ReLU(True),
        )
        self.ups, self.upc = nn.ModuleList(), nn.ModuleList()
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose3d(f*2, f, 2, stride=2))
            self.upc.append(nn.Sequential(
                nn.Conv3d(f*2, f, 3, padding=1),
                nn.BatchNorm3d(f),
                nn.ReLU(True),
                nn.Conv3d(f, f, 3, padding=1),
                nn.BatchNorm3d(f),
                nn.ReLU(True),
            ))
        self.final = nn.Conv3d(feats[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, conv, skip in zip(self.ups, self.upc, reversed(skips)):
            x = up(x)
            # 對齊尺寸
            diff = [skip.size(i) - x.size(i) for i in range(2,5)]
            pad = []
            for d in reversed(diff): pad += [0, max(0,d)]
            if any(d < 0 for d in diff):
                crop = [slice(0, x.size(i) + min(0,d)) for i,d in enumerate(diff, start=2)]
                x = x[:, :, crop[0], crop[1], crop[2]]
            x = nn.functional.pad(x, pad)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)
        return self.final(x)

# 單塊推論，用於 dask.map_blocks
def inference_block(block, model, device, threshold, target_shape):
    orig = block.shape
    pad_w = [(0, target_shape[i] - orig[i]) for i in range(3)]
    p = np.pad(block.astype(np.float32), pad_w, mode='constant', constant_values=0)
    inp = torch.from_numpy(p[None,None]).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()
    mask_full = (probs > threshold).astype(np.uint8)
    return mask_full[:orig[0], :orig[1], :orig[2]]

# 處理單檔案
def process_file(tif_path, model, device, dz, dy, dx, threshold, output_folder):
    vol = tif_memmap(tif_path)
    chunks = (dz, dy, dx)
    dvol = da.from_array(vol, chunks=chunks)
    wrapped = da.map_blocks(
        inference_block,
        dvol,
        model,
        device,
        threshold,
        chunks,
        dtype=np.uint8,
        chunks=chunks
    )
    # 顯示 chunk 推論進度
    with ProgressBar():
        mask = wrapped.compute()
    out_name = os.path.basename(tif_path)
    out_path = os.path.join(output_folder, out_name)
    imwrite(out_path, mask, dtype=np.uint8, compression='zlib')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in',  dest='input_folder',  required=True, help='3D輸入 TIFF 資料夾')
    parser.add_argument('--m',   dest='model_path',     required=True, help='model.pth 路徑')
    parser.add_argument('--o',   dest='output_folder',  required=True, help='輸出 mask 資料夾')
    parser.add_argument('--dz',  type=int, default=16, help='Patch 深度')
    parser.add_argument('--dy',  type=int, default=48, help='Patch 高度')
    parser.add_argument('--dx',  type=int, default=48, help='Patch 寬度')
    parser.add_argument('--threshold', type=float, default=0.5, help='機率門檻')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet3D().to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.input_folder, '*.tif')))
    # 顯示檔案處理進度
    for tif in tqdm(files, desc='Processing files'):
        process_file(tif, model, device,
                     args.dz, args.dy, args.dx,
                     args.threshold,
                     args.output_folder)
