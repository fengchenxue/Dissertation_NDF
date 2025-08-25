import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import model_L

@torch.no_grad()
def eval_l2lowrank(npz_path, ckpt_path, device=None):
    import torch, numpy as np
    from model_G import L2LowRankHead
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    d = np.load(npz_path)
    z = torch.from_numpy(d['z'][None]).float().to(device)     # (1,z_dim)
    I = torch.from_numpy(d['i_angles']).float().to(device)     # (N_i,4)
    L = torch.from_numpy(d['L2p']).float().to(device)          # (N_i,H,W)
    w = torch.from_numpy(d['weight_o']).float().to(device)     # (H,W)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = L2LowRankHead(**ckpt['cfg']).to(device)
    model.load_state_dict(ckpt['model']); model.eval()

    B=256; maes=[]; rmses=[]; p95s=[]
    for s in range(0, I.shape[0], B):
        zB = z.expand(min(B, I.shape[0]-s), -1)
        Lp,_ = model.forward_reconstruct(I[s:s+B], z=zB)       # (B,H,W)
        diff = torch.abs(Lp - L[s:s+B])
        mae  = (diff * w).mean(dim=(-1,-2))
        rmse = torch.sqrt(((diff**2) * w).mean(dim=(-1,-2)))
        p95  = torch.quantile((diff*w).flatten(1), 0.95, dim=1)
        maes.append(mae.cpu()); rmses.append(rmse.cpu()); p95s.append(p95.cpu())
    mae = torch.cat(maes).mean().item()
    rmse= torch.cat(rmses).mean().item()
    p95 = torch.cat(p95s).mean().item()
    print(f"[L2LowRank][{os.path.basename(npz_path)}]  MAE={mae:.6f}  RMSE={rmse:.6f}  P95={p95:.6f}")

