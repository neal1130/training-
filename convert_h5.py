#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, numpy as np, tifffile, h5py
from tqdm import tqdm

raw_dir  = r'D:\kun\spine\HiNGE\raw'
mask_dir = r'D:\kun\spine\HiNGE\mask'
h5_path  = r'D:\kun\spine\HiNGE\HiNGe_dataset.h5'

vol_paths = sorted(glob.glob(os.path.join(raw_dir,  '*.tif')))
msk_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))

# 删除旧文件，创建空 HDF5
if os.path.exists(h5_path):
    os.remove(h5_path)
with h5py.File(h5_path, 'w'):
    pass

# 串行写入
with h5py.File(h5_path, 'a') as f:
    for i, (vp, mp) in enumerate(tqdm(zip(vol_paths, msk_paths),
                                       total=len(vol_paths),
                                       desc="Writing HDF5")):
        vol = tifffile.imread(vp)
        msk = tifffile.imread(mp)
        if vol.ndim == 2:
            vol = vol[np.newaxis, ...]
            msk = msk[np.newaxis, ...]
        grp = f.create_group(f'subject_{i}')
        grp.create_dataset('vol', data=vol, compression='lzf')
        grp.create_dataset('msk', data=msk, compression='lzf')
