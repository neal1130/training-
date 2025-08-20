# dataset_h5.py
import h5py, numpy as np, torch
from torch.utils.data import Dataset
from tqdm import tqdm

class HDF5PatchDataset(Dataset):
    def __init__(self, h5_path, items, patch_size, rotate: bool = False):
        self.h5_path = h5_path
        self.items   = items
        self.dz, self.dy, self.dx = patch_size
        self.h5 = None
        self.rotate = rotate
        # 只在 XY 平面旋轉
        self.plane_axes = [(2,3)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, 'r')
        subj, (z,y,x), _ = self.items[idx]
        grp = self.h5[subj]
        vol = grp['vol'][z:z+self.dz, y:y+self.dy, x:x+self.dx]
        msk = grp['msk'][z:z+self.dz, y:y+self.dy, x:x+self.dx]

        # zero-pad
        pad = [(0, max(0, self.dz - vol.shape[0])),
               (0, max(0, self.dy - vol.shape[1])),
               (0, max(0, self.dx - vol.shape[2]))]
        if any(p[1]>0 for p in pad):
            vol = np.pad(vol, pad, mode='constant', constant_values=0)
            msk = np.pad(msk, pad, mode='constant', constant_values=0)

        img  = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)
        mask = torch.from_numpy((msk>0).astype(np.float32)).unsqueeze(0)
        return img, mask

# 加入 tqdm 顯示進度
def enumerate_patches_h5(h5_path, patch_size, overlap):
    dz, dy, dx = patch_size
    oz, oy, ox = overlap
    step_z, step_y, step_x = dz-oz, dy-oy, dx-ox

    # 打開檔案並增大 chunk cache
    f = h5py.File(
        h5_path, 'r',
        rdcc_nbytes=512*1024**2,
        rdcc_nslots=1_000_000
    )

    items = []
    for subj in tqdm(f.keys(), desc="Enumerating volumes"):
        # 1) 直接把整個 mask 讀進記憶體，轉成 NumPy 陣列
        msk = f[subj]['msk'][...]    # shape = (D, H, W)
        D, H, W = msk.shape

        # 2) 向量化生成所有起點
        zs = np.concatenate([np.arange(0, D-dz+1, step_z), [D-dz]])
        ys = np.concatenate([np.arange(0, H-dy+1, step_y), [H-dy]])
        xs = np.concatenate([np.arange(0, W-dx+1, step_x), [W-dx]])
        grid = np.stack(np.meshgrid(zs, ys, xs, indexing='ij'), -1).reshape(-1, 3)  # (N,3)

        # 3) 生成可廣播的索引，shape 最終變成 (N, dz, dy, dx)
        z_idx = grid[:,0][:, None, None, None] + np.arange(dz)[None, :, None, None]
        y_idx = grid[:,1][:, None, None, None] + np.arange(dy)[None, None, :, None]
        x_idx = grid[:,2][:, None, None, None] + np.arange(dx)[None, None, None, :]

        # 4) 在 NumPy 陣列上做一次性 fancy indexing
        patches = msk[z_idx, y_idx, x_idx]          # (N, dz, dy, dx)
        labels  = np.any(patches > 0, axis=(1,2,3)) # (N,)

        # 5) 組裝 items 列表
        items.extend([
            (subj, (int(z), int(y), int(x)), int(lbl))
            for (z, y, x), lbl in zip(grid, labels)
        ])

    f.close()
    return items



def sample_half_foreground_balanced(items):
    import random
    from tqdm import tqdm
    pos = [i for i in items if i[2]==1]
    neg = [i for i in items if i[2]==0]
    k = max(1, len(pos))
    k1 = max(1, len(pos)//2)
    sel = []
    for _ in tqdm(range(k), desc="Sampling positives"):
        sel.append(random.choice(pos))
    for _ in tqdm(range(k1), desc="Sampling negatives"):
        sel.append(random.choice(neg))
    random.shuffle(sel)
    return sel
